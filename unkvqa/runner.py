"""Generation engine for the Qwen graded probe.

Generation and scoring remain separate so saved JSONL outputs can be rescored.
Batched generation uses left padding, while forced-choice scoring uses right
padding and an explicit shared-prefix length. Image resolution is fixed for
reproducibility, and runs resume by item, condition, protocol, quantization,
and prompt variant.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from . import protocols  # noqa: E402
from .data import Condition, Item, image_path  # noqa: E402
from .protocols import MAX_NEW_TOKENS, Prompt, Protocol  # noqa: E402

# Qwen2.5-VL defaults, fixed for reproducibility.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

SEED = 0

# Per-level batch sizes tuned for the 8 GB evaluation GPU. Bf16 forced-choice
# scoring stays serial to avoid WDDM host-memory spill.
BATCH_DEFAULTS = {"bf16": 2, "int8": 4, "int4": 8}
CAND_BATCH_DEFAULTS = {"bf16": 1, "int8": 2, "int4": 4}


def build_quant_config(quant: str):
    """Return the weight-loading configuration used in the paper."""
    if quant == "bf16":
        return None, torch.bfloat16
    if quant == "int8":
        return (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head"],
            ),
            torch.bfloat16,
        )
    if quant == "int4":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch.bfloat16,
        )
    raise ValueError(f"unsupported quantization level: {quant}")


def record_key(rec: dict) -> tuple:
    """Resume identity. `variant` defaults to the standard prompt so JSONLs
    written before prompt variants existed still resume correctly."""
    return (
        rec["item_id"], rec["condition"], rec["protocol"], rec["quant"],
        rec.get("variant", protocols.DEFAULT_VARIANT),
    )


def load_done_keys(path: str) -> set[tuple]:
    """Keys already present in an existing JSONL, for resume."""
    done: set[tuple] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a hard kill
    return done


@dataclass
class Task:
    item: Item
    condition: str
    protocol: Protocol


#: UNK-VQA has two conditions; the constructed probe has three
#: (original / target / control). Parameterised rather than hardcoded so one
#: runner drives both datasets.
DEFAULT_CONDITIONS: tuple[str, ...] = ("original", "masked")


def build_tasks(
    items: Iterable[Item],
    protos: Iterable[Protocol],
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
) -> list[Task]:
    conditions = tuple(conditions)
    return [
        Task(item=it, condition=cond, protocol=proto)
        for it in items
        for cond in conditions
        for proto in protos
    ]


def _chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Engine:
    """One loaded model at one quantization level."""

    def __init__(self, model_path: str, quant: str, device: str = "cuda:0"):
        self.quant = quant
        self.device = device
        torch.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()

        quant_config, dtype = build_quant_config(quant)
        t0 = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            quantization_config=quant_config,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        self.load_s = time.time() - t0
        self.weights_gb = torch.cuda.memory_allocated() / 1e9

    # -- shared plumbing -------------------------------------------------

    def _prepare(self, messages_batch: list[list[dict]], texts: list[str], padding_side: str):
        self.processor.tokenizer.padding_side = padding_side
        images, videos = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts, images=images, videos=videos, padding=True, return_tensors="pt"
        )
        return inputs.to(self.device)

    def _chat_text(self, messages: list[dict]) -> str:
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # -- freeform --------------------------------------------------------

    @torch.inference_mode()
    def generate_freeform(self, prompts: list[Prompt]) -> list[str]:
        """Batched greedy generation. LEFT padding (note 1)."""
        messages_batch = [p.messages for p in prompts]
        texts = [self._chat_text(p.messages) for p in prompts]
        inputs = self._prepare(messages_batch, texts, padding_side="left")

        out = self.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return [
            t.strip()
            for t in self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]

    # -- forced choice ---------------------------------------------------

    @torch.inference_mode()
    def score_candidates(self, prompt: Prompt, cand_batch: int = 4) -> dict[str, dict[str, float]]:
        """Log-probability of each candidate as a continuation of the same prefix.

        Returns {candidate_key: {"text", "logprob_sum", "logprob_mean", "n_tokens"}}.

        BOTH sum and length-normalised mean are recorded, and the scorer decides.
        This is not indecision: the smoke test showed the choice matters a lot.
        "unanswerable" tokenizes to 3 pieces whose trailing tokens are nearly
        certain once the first is emitted, so the MEAN is dragged toward zero and
        systematically beats 1-token content answers even when the model plainly
        knows them (agreement with freeform on originals: sum 5/6, mean 1/6).
        That is surface-form competition (Holtzman et al. 2021). Keeping both
        here means the decision rule can be settled on evidence, offline, without
        regenerating anything.
        """
        assert prompt.candidates is not None
        cands = list(prompt.candidates)
        prefix_text = self._chat_text(prompt.messages)

        # Prefix length, measured once. Identical across candidates because they
        # share an image and a question (note 2).
        prefix_inputs = self._prepare([prompt.messages], [prefix_text], padding_side="right")
        prefix_len = int(prefix_inputs.attention_mask[0].sum().item())

        out: dict[str, dict[str, float]] = {}
        for chunk in _chunks(cands, max(1, cand_batch)):
            messages_batch = [prompt.messages] * len(chunk)
            texts = [prefix_text + c.text for c in chunk]
            inputs = self._prepare(messages_batch, texts, padding_side="right")

            logits = self.model(**inputs).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)

            for i, cand in enumerate(chunk):
                real_len = int(inputs.attention_mask[i].sum().item())
                n_tokens = real_len - prefix_len
                if n_tokens <= 0:  # candidate tokenized away to nothing
                    out[cand.key] = {
                        "text": cand.text, "logprob_sum": float("-inf"),
                        "logprob_mean": float("-inf"), "n_tokens": 0,
                    }
                    continue
                # Token at position p is predicted by the logits at position p-1.
                positions = torch.arange(prefix_len, real_len, device=logits.device)
                target_ids = inputs.input_ids[i, positions]
                token_lp = (
                    logprobs[i, positions - 1, :]
                    .gather(-1, target_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
                total = float(token_lp.sum().item())
                out[cand.key] = {
                    "text": cand.text,
                    "logprob_sum": total,
                    "logprob_mean": total / n_tokens,
                    "n_tokens": n_tokens,
                }
            del logits, logprobs, inputs
        return out


def run(
    items: list[Item],
    dataset_root: str,
    model_path: str,
    quant_levels: Iterable[str],
    protos: Iterable[Protocol],
    out_path: str,
    batch_size: int | None = None,
    cand_batch: int | None = None,
    resume: bool = True,
    variant: str = protocols.DEFAULT_VARIANT,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
    path_fn=image_path,
) -> None:
    """Run every (item, condition, protocol) at every quant level, appending JSONL.

    `conditions` and `path_fn` are parameterised so the same runner serves both
    UNK-VQA (original/masked) and the constructed probe (original/target/control).
    """
    conditions = tuple(conditions)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = load_done_keys(out_path) if resume else set()
    if done:
        print(f"[resume] {len(done)} records already present in {out_path}")

    protos = list(protos)
    run_id = f"{int(time.time())}"

    for quant in quant_levels:
        tasks = [
            t for t in build_tasks(items, protos, conditions)
            if (t.item.item_id, t.condition, t.protocol, quant, variant) not in done
        ]
        if not tasks:
            print(f"[{quant}] nothing to do")
            continue

        bs = batch_size or BATCH_DEFAULTS.get(quant, 4)
        cb = cand_batch or CAND_BATCH_DEFAULTS.get(quant, 2)

        print(f"[{quant}] loading model...")
        engine = Engine(model_path, quant)
        print(f"[{quant}] loaded in {engine.load_s:.1f}s  weights={engine.weights_gb:.2f} GB  "
              f"tasks={len(tasks)}  batch={bs} cand_batch={cb}")

        free = [t for t in tasks if t.protocol == "freeform"]
        forced = [t for t in tasks if t.protocol == "forced_choice"]

        t_start = time.time()
        n_done = 0
        with open(out_path, "a", encoding="utf-8") as fh:

            def emit(task: Task, **extra) -> None:
                nonlocal n_done
                it = task.item
                if hasattr(it, "record_fields"):  # constructed probe
                    meta = it.record_fields(task.condition)
                else:  # UNK-VQA
                    meta = {
                        "split": it.split, "question_id": it.question_id,
                        "question": it.question,
                        "answerable": it.answerable, "reason": it.reason,
                        "gold_orig": it.gold_original, "gold_masked": it.gold_masked,
                        "abstain_code": it.abstain_code, "options": it.options,
                        "should_abstain": it.should_abstain(task.condition),
                        # Explicit per-condition gold so the scorer never has to
                        # re-derive which field applies to which condition.
                        "gold": it.gold_for(task.condition),
                    }
                rec = {
                    "item_id": it.item_id, "condition": task.condition,
                    "image_file": it.image_for(task.condition),
                    **meta,
                    "quant": quant, "protocol": task.protocol, "variant": variant,
                    "run_id": run_id, **extra,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1

            def progress(tag: str, i: int, total: int) -> None:
                el = time.time() - t_start
                rate = n_done / max(el, 1e-6)
                eta = (len(tasks) - n_done) / max(rate, 1e-6)
                print(f"  [{quant}/{tag}] {i}/{total}  ({rate:.1f} rec/s, eta {eta/60:.0f}m)")

            # freeform: batched generation
            for bi, batch in enumerate(_chunks(free, bs), 1):
                prompts = [
                    protocols.build(t.item, t.condition, "freeform",
                                    path_fn(dataset_root, t.item.image_for(t.condition)),
                                    variant)
                    for t in batch
                ]
                t0 = time.time()
                outputs = engine.generate_freeform(prompts)
                dt = (time.time() - t0) / len(batch)
                for task, text in zip(batch, outputs):
                    emit(task, raw_output=text, option_logprobs=None, gen_s=round(dt, 3))
                fh.flush()
                if bi % 10 == 0 or bi * bs >= len(free):
                    progress("freeform", min(bi * bs, len(free)), len(free))

            # Generation leaves the caching allocator full and fragmented. Without
            # this, int4 forced_choice OOMed immediately after int4 freeform
            # finished: the SMALLEST model, which is the tell that it was
            # fragmentation rather than model size.
            if free:
                torch.cuda.empty_cache()

            # forced_choice: one scoring pass per item (candidates batched inside)
            oom_skipped = 0
            for i, task in enumerate(forced, 1):
                prompt = protocols.build(
                    task.item, task.condition, "forced_choice",
                    path_fn(dataset_root, task.item.image_for(task.condition)),
                    variant,
                )
                t0 = time.time()
                try:
                    scores = engine.score_candidates(prompt, cand_batch=cb)
                except torch.AcceleratorError:
                    # Retry once, serialised. An unattended run must not die on a
                    # transient allocation failure after hours of work; anything
                    # still failing is left unwritten so resume picks it up later.
                    torch.cuda.empty_cache()
                    try:
                        scores = engine.score_candidates(prompt, cand_batch=1)
                    except torch.AcceleratorError:
                        oom_skipped += 1
                        print(f"  [{quant}/forced] OOM on {task.item.item_id}"
                              f"/{task.condition}, skipped (resume will retry)")
                        torch.cuda.empty_cache()
                        continue
                emit(task, raw_output=None, option_logprobs=scores,
                     gen_s=round(time.time() - t0, 3))
                if i % 100 == 0 or i == len(forced):
                    fh.flush()
                    progress("forced", i, len(forced))
            fh.flush()
            if oom_skipped:
                print(f"  [{quant}/forced] {oom_skipped} records skipped on OOM "
                      f": re-run to fill them in")

        peak = torch.cuda.max_memory_allocated() / 1e9
        el = time.time() - t_start
        print(f"[{quant}] done: {n_done} records in {el/60:.1f}m  peak_vram={peak:.2f} GB")

        del engine
        torch.cuda.empty_cache()
