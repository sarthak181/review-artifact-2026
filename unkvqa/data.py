"""Load paired original and masked examples from the UNK-VQA I-2 schema.

`answerability.other.answer` contains either a literal answer or an answer-map
code, depending on `binary`. The loader keeps those cases in `gold_masked` and
`abstain_code`. Original and masked answers are stored separately and selected
through `Item.gold_for(condition)`. Pair validation is strict by default so
missing images cannot silently reduce the evaluation set.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Literal

Condition = Literal["original", "masked"]

# From the annotation files' own top-level maps, reproduced for reference.
ANSWER_MAP = {
    "1": "I don't know (e.g., beyond my knowledge)",
    "2": "Not sure (e.g., multiple answers)",
    "3": "I cannot answer (e.g., difficult question)",
}
REASON_MAP = {
    "1": "It has multiple plausible answers",
    "2": "It is difficult to understand",
    "3": "The image lacks important concepts/information",
    "4": "It requires higher-level knowledge to answer",
}
#: `reason` code marking a visually-grounded abstention. Only ~24.7% of
#: unanswerable I-2 items carry it, which is why groundedness is measured
#: behaviourally rather than read off this field.
REASON_IMAGE_LACKS_INFO = "3"

ALTER_TYPE = "I-2"
IMAGES_SUBDIR = "images-val"


@dataclass(frozen=True)
class Item:
    """One I-2 probe item, with both of its images."""

    item_id: str  # "val:97462": unique across splits
    split: str
    question_id: int
    question: str

    answerable: bool  # True -> should answer; False -> should abstain
    reason: str | None  # reason_map code; unanswerable items only

    masked_file: str
    original_file: str

    gold_original: str  # misc.answer_origin: correct answer BEFORE masking
    gold_masked: str | None  # other.answer: answerable items only
    abstain_code: str | None  # answer_map code: unanswerable items only

    options: dict[str, str] = field(default_factory=dict)  # orig/baseline/random

    def image_for(self, condition: Condition) -> str:
        return self.original_file if condition == "original" else self.masked_file

    def gold_for(self, condition: Condition) -> str | None:
        """Correct answer for this condition, or None when the model should abstain.

        None is meaningful: it means "no answer is correct here, abstention is
        the right behaviour". Only ever returned for masked+unanswerable.
        """
        if condition == "original":
            return self.gold_original
        return self.gold_masked if self.answerable else None

    def should_abstain(self, condition: Condition) -> bool:
        """Ground-truth behaviour. Originals are always answerable by construction."""
        return condition == "masked" and not self.answerable


def _parse_entry(entry: dict, split: str) -> Item | None:
    if entry.get("alter_type") != ALTER_TYPE:
        return None

    ans = entry["answerability"]
    other = ans["other"]
    binary = bool(ans["binary"])
    misc = entry.get("misc", {})

    return Item(
        item_id=f"{split}:{entry['question_id']}",
        split=split,
        question_id=entry["question_id"],
        question=entry["question"],
        answerable=binary,
        reason=None if binary else other.get("reason"),
        masked_file=entry["image_name"],
        original_file=misc["image_name_origin"],
        gold_original=misc["answer_origin"],
        # The polymorphism, resolved once, here.
        gold_masked=other["answer"] if binary else None,
        abstain_code=None if binary else other["answer"],
        options=dict(other.get("options", {})),
    )


def load_items(
    root: str,
    splits: Iterable[str] = ("val", "test"),
    require_pairs: bool = True,
) -> list[Item]:
    """Load all I-2 items for the given splits.

    `require_pairs` raises if any item is missing either image, rather than
    letting the evaluation quietly run on a subset.
    """
    img_dir = os.path.join(root, IMAGES_SUBDIR)
    on_disk = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()

    items: list[Item] = []
    for split in splits:
        path = os.path.join(root, f"annt_{split}.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data["annotation"]:
            item = _parse_entry(entry, split)
            if item is not None:
                items.append(item)

    missing_mask = [i for i in items if i.masked_file not in on_disk]
    missing_orig = [i for i in items if i.original_file not in on_disk]
    if require_pairs and (missing_mask or missing_orig):
        raise FileNotFoundError(
            f"{len(missing_mask)} masked and {len(missing_orig)} original images "
            f"missing from {img_dir}. Run:\n"
            f"  python download_i2.py --include-originals\n"
            f"  python download_coco_originals.py\n"
            f"e.g. missing original: {missing_orig[0].original_file if missing_orig else '-'}"
        )
    if not require_pairs:
        items = [
            i for i in items
            if i.masked_file in on_disk and i.original_file in on_disk
        ]
    return items


def image_path(root: str, filename: str) -> str:
    return os.path.join(root, IMAGES_SUBDIR, filename)


#: Perturbation types that alter the IMAGE, leaving the question text untouched.
#: T-1/T-2 rewrite the question and the original wording is not stored, so their
#: entries cannot be recovered as clean question/answer pairs.
IMAGE_ALTER_TYPES = ("I-1", "I-2", "I-3")


@dataclass(frozen=True)
class OriginQA:
    """An unperturbed question with its verified pre-perturbation answer."""

    key: str  # "<origin_question_id>:<origin_image>"
    split: str
    question: str
    answer: str  # misc.answer_origin
    image_file: str  # misc.image_name_origin
    alter_types: tuple[str, ...]  # which perturbations were derived from it


def load_origin_questions(root: str, splits: Iterable[str] = ("val", "test")) -> list[OriginQA]:
    """Recover the clean (question, answer, original image) triples underlying
    every image-perturbation entry, deduplicated.

    Used to build a larger pool for the constructed-mask probe than I-2 alone
    provides: the same origin question is often perturbed several ways, and all
    we need from it is the unperturbed question and its true answer.
    """
    seen: dict[str, dict] = {}
    for split in splits:
        path = os.path.join(root, f"annt_{split}.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for e in data["annotation"]:
            if e.get("alter_type") not in IMAGE_ALTER_TYPES:
                continue
            misc = e.get("misc", {})
            origin_img = misc.get("image_name_origin")
            answer = misc.get("answer_origin")
            if not origin_img or not answer:
                continue
            key = f"{misc.get('question_id_origin')}:{origin_img}"
            rec = seen.setdefault(key, {
                "key": key, "split": split, "question": e["question"],
                "answer": answer, "image_file": origin_img, "alter": set(),
            })
            rec["alter"].add(e["alter_type"])
    return [
        OriginQA(key=r["key"], split=r["split"], question=r["question"],
                 answer=r["answer"], image_file=r["image_file"],
                 alter_types=tuple(sorted(r["alter"])))
        for r in seen.values()
    ]


def summarize(items: list[Item]) -> str:
    """Human-readable breakdown: used by the __main__ self-check."""
    import collections

    lines = [f"items: {len(items)}"]
    by_split = collections.Counter(i.split for i in items)
    lines.append(f"  by split: {dict(sorted(by_split.items()))}")

    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    lines.append(f"  answerable (should answer):   {len(answerable)}")
    lines.append(f"  unanswerable (should abstain): {len(unanswerable)}")

    reasons = collections.Counter(i.reason for i in unanswerable)
    lines.append(f"  unanswerable reason codes: {dict(sorted(reasons.items(), key=lambda kv: str(kv[0])))}")
    grounded = sum(1 for i in unanswerable if i.reason == REASON_IMAGE_LACKS_INFO)
    lines.append(
        f"    reason=3 (image lacks info): {grounded} "
        f"({grounded / max(len(unanswerable), 1):.1%} of unanswerable)"
    )

    # Sanity: the gold answer for a masked answerable item is one of its options.
    off_option = [
        i for i in answerable
        if i.gold_masked is not None and i.gold_masked not in i.options.values()
    ]
    lines.append(f"  answerable items whose gold is NOT among options: {len(off_option)}")

    agree = sum(1 for i in answerable if i.gold_masked == i.gold_original)
    lines.append(
        f"  gold_masked == gold_original: {agree}/{len(answerable)} "
        f"({agree / max(len(answerable), 1):.1%}): why gold_for(condition) exists"
    )
    lines.append(f"  unique original images: {len({i.original_file for i in items})}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Self-check the I-2 loader.")
    p.add_argument("--root", default=os.path.join(here, "dataset", "UNK-VQA"))
    p.add_argument("--splits", default="val,test")
    args = p.parse_args()

    loaded = load_items(args.root, [s for s in args.splits.split(",") if s])
    print(summarize(loaded))
    print("\nsample item:")
    sample = next(i for i in loaded if not i.answerable)
    for k, v in sample.__dict__.items():
        print(f"  {k}: {v}")
    print(f"  gold_for(original) = {sample.gold_for('original')!r}")
    print(f"  gold_for(masked)   = {sample.gold_for('masked')!r}  (None = should abstain)")
