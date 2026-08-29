"""
Loader for the constructed-mask probe (see build_probe.py).

Deliberately mirrors the `Item` interface from data.py: `image_for`,
`gold_for`, `should_abstain`: so the same runner drives both datasets. The only
structural difference is a third condition:

    original  no mask                                -> answer
    target    the ANSWER object removed              -> abstain
    control   a DIFFERENT object of matched area removed -> answer

`control` is what UNK-VQA I-2 never really had: a mask of comparable size that
leaves the evidence intact. It is what makes the false-positive rate mean
something, and therefore what makes GAI mean something.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

ProbeCondition = Literal["original", "target", "control"]
PROBE_CONDITIONS: tuple[ProbeCondition, ...] = ("original", "target", "control")

MANIFEST = "manifest.json"
IMAGES_SUBDIR = "images"


@dataclass(frozen=True)
class ProbeItem:
    item_id: str
    split: str
    question: str
    answer: str  # correct answer whenever the evidence is present

    image_original: str
    image_target: str
    image_control: str

    target_category: str
    control_category: str
    target_masked_frac: float
    control_masked_frac: float
    source_image: str

    def image_for(self, condition: ProbeCondition) -> str:
        return {
            "original": self.image_original,
            "target": self.image_target,
            "control": self.image_control,
        }[condition]

    def gold_for(self, condition: ProbeCondition) -> str | None:
        """None means 'no answer is correct here': abstention is the right move.
        Only the target condition returns None, and only because that is the one
        condition where the answer-bearing object has been removed."""
        return None if condition == "target" else self.answer

    def should_abstain(self, condition: ProbeCondition) -> bool:
        return condition == "target"

    def record_fields(self, condition: ProbeCondition) -> dict:
        """Metadata written into every JSONL record for this (item, condition)."""
        return {
            "split": self.split,
            "question": self.question,
            "answer": self.answer,
            "gold": self.gold_for(condition),
            "should_abstain": self.should_abstain(condition),
            "target_category": self.target_category,
            "control_category": self.control_category,
            "target_masked_frac": self.target_masked_frac,
            "control_masked_frac": self.control_masked_frac,
            "source_image": self.source_image,
        }


#: Graded probe occlusion levels: calibrated so that true object-pixel coverage
#: is evenly spaced at 20/40/60/80/95/100%. See build_graded.py for the mapping.
GRADED_LEVELS = (11, 23, 37, 54, 73, 100)
GRADED_CONDITIONS: tuple[str, ...] = (
    ("original",) + tuple(f"target_{l}" for l in GRADED_LEVELS) + ("control",)
)


@dataclass(frozen=True)
class GradedItem:
    """One question under graded occlusion of its answer object.

    Only `target_100` has a definite ground truth of 'abstain'. The partial
    levels are deliberately UNLABELLED: what the model should do at 50%
    occlusion is the empirical question, not something to assert in advance.
    So `should_abstain` is False for them and they contribute to the curve,
    never to a correctness score.
    """

    item_id: str
    split: str
    question: str
    answer: str
    images: dict  # condition -> filename
    masked_frac: dict  # condition -> painted fraction
    target_category: str
    control_category: str
    source_image: str

    def image_for(self, condition: str) -> str:
        return self.images[condition]

    def gold_for(self, condition: str) -> str | None:
        return None if condition == "target_100" else self.answer

    def should_abstain(self, condition: str) -> bool:
        return condition == "target_100"

    def record_fields(self, condition: str) -> dict:
        return {
            "split": self.split, "question": self.question, "answer": self.answer,
            "gold": self.gold_for(condition),
            "should_abstain": self.should_abstain(condition),
            "occlusion": (int(condition.split("_")[1])
                          if condition.startswith("target_") else
                          (0 if condition == "original" else None)),
            "masked_frac": self.masked_frac.get(condition),
            "target_category": self.target_category,
            "control_category": self.control_category,
            "source_image": self.source_image,
        }


def load_graded(root: str, limit: int = 0, seed: int = 0) -> list[GradedItem]:
    with open(os.path.join(root, MANIFEST), encoding="utf-8") as fh:
        raw = json.load(fh)
    items = []
    for r in raw:
        images = {"original": r["image_original"], "control": r["image_control"]}
        fracs = {"original": 0.0, "control": r["control_masked_frac"]}
        for l in GRADED_LEVELS:
            images[f"target_{l}"] = r[f"image_target_{l}"]
            fracs[f"target_{l}"] = r[f"masked_frac_{l}"]
        items.append(GradedItem(
            item_id=r["item_id"], split=r.get("split", "val"), question=r["question"],
            answer=r["answer"], images=images, masked_frac=fracs,
            target_category=r["target_category"],
            control_category=r["control_category"],
            source_image=r["source_image"],
        ))
    if limit and limit < len(items):
        import random
        random.Random(seed).shuffle(items)
        items = items[:limit]
        items.sort(key=lambda i: i.item_id)
    return items


def load_probe(root: str, limit: int = 0, seed: int = 0) -> list[ProbeItem]:
    """Load the manifest. `limit` takes a deterministic random subset, not a
    prefix: manifest order follows COCO image id, which correlates with nothing
    useful but is not random either."""
    with open(os.path.join(root, MANIFEST), encoding="utf-8") as fh:
        raw = json.load(fh)
    items = [
        ProbeItem(
            item_id=r["item_id"], split=r.get("split", "val"), question=r["question"],
            answer=r["answer"], image_original=r["image_original"],
            image_target=r["image_target"], image_control=r["image_control"],
            target_category=r["target_category"], control_category=r["control_category"],
            target_masked_frac=r["target_masked_frac"],
            control_masked_frac=r["control_masked_frac"],
            source_image=r["source_image"],
        )
        for r in raw
    ]
    if limit and limit < len(items):
        import random
        random.Random(seed).shuffle(items)
        items = items[:limit]
        items.sort(key=lambda i: i.item_id)
    return items


def image_path(root: str, filename: str) -> str:
    return os.path.join(root, IMAGES_SUBDIR, filename)
