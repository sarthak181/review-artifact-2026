"""
Cross-FAMILY, MID-SCALE generality: run the graded probe on Gemma-3-4B-it
(google, ~4.3B, gated; Gemma-3 LLM + SigLIP vision -- no Qwen) at int8. Fills the
middle of the 3B(Qwen) -> 4B(Gemma) -> 7B(Qwen) scale ladder.

Native transformers class (Gemma3ForConditionalGeneration) -- no remote code, so
it works cleanly on transformers 5.x (unlike Phi-3.5-vision). Emits the SAME JSONL
schema as run_graded.py (reuses unkvqa.probe + protocols) so score.py /
graded_eval.py work unchanged. Single process, resumable.

Requires HF access: accept the license at hf.co/google/gemma-3-4b-it and
`huggingface-cli login` (read token). No token handled in code.

  python run_graded_gemma.py --variant standard --limit 3   # smoke
  python run_graded_gemma.py --variant standard             # full 2400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unkvqa.probe import GRADED_CONDITIONS, image_path, load_graded  # noqa: E402
from unkvqa.protocols import MAX_NEW_TOKENS, PROMPT_VARIANTS  # noqa: E402

MODEL_ID = "google/gemma-3-4b-it"
PROBE_ROOT = os.path.join(HERE, "dataset", "probe_graded")
OUT = os.path.join(HERE, "results", "graded_gemma.jsonl")
QUANT = "int8"


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["condition"], r["protocol"], r["quant"],
                          r.get("variant", "standard")))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="standard", choices=sorted(PROMPT_VARIANTS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    items = load_graded(PROBE_ROOT, limit=args.limit)
    done = load_done(args.out)
    tasks = [(it, c) for it in items for c in GRADED_CONDITIONS
             if (it.item_id, c, "freeform", QUANT, args.variant) not in done]
    if not tasks:
        print(f"[gemma/{args.variant}] nothing to do ({len(done)} present)")
        return
    print(f"[gemma/{args.variant}] {len(tasks)} tasks ({len(done)} done)")

    torch.cuda.reset_peak_memory_stats()
    # int8; keep SigLIP vision tower + projector + lm_head in fp16.
    qcfg = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
    t0 = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, quantization_config=qcfg,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[gemma] loaded {MODEL_ID} in {time.time()-t0:.1f}s "
          f"weights={torch.cuda.memory_allocated()/1e9:.2f} GB")

    run_id = str(int(time.time()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start, n = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as fh:
        for it, cond in tasks:
            img = Image.open(image_path(PROBE_ROOT, it.image_for(cond))).convert("RGB")
            text_prompt = f"{PROMPT_VARIANTS[args.variant]}\n\nQuestion: {it.question}"
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text_prompt}]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            g0 = time.time()
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            raw = processor.decode(gen[0][input_len:], skip_special_tokens=True).strip()
            rec = {"item_id": it.item_id, "condition": cond,
                   "image_file": it.image_for(cond), **it.record_fields(cond),
                   "quant": QUANT, "protocol": "freeform", "variant": args.variant,
                   "model": MODEL_ID, "run_id": run_id,
                   "raw_output": raw, "option_logprobs": None,
                   "gen_s": round(time.time() - g0, 3)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                fh.flush()
                rate = n / max(time.time() - start, 1e-6)
                print(f"  [{args.variant}] {n}/{len(tasks)} ({rate:.2f} rec/s, "
                      f"eta {(len(tasks)-n)/max(rate,1e-6)/60:.0f}m)")
        fh.flush()
    print(f"[gemma] done: {n} records in {(time.time()-start)/60:.1f}m "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
