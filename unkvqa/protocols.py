"""Prompt construction for free-form and forced-choice evaluation.

Forced choice compares candidate continuation log probabilities and avoids
parsing generated text. Free-form generation supports analysis of refusal
wording but is more sensitive to response format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .data import Condition, Item

Protocol = Literal["freeform", "forced_choice"]

#: The string the model is told to use, and the abstain candidate in forced_choice.
ABSTAIN_TEXT = "unanswerable"

#: Candidate provenance keys. `abstain` is ours; the rest come from `options`.
CANDIDATE_KEYS = ("orig", "baseline", "random", "abstain")

#: Prompt variants, ordered from least to most abstention-encouraging.
#:
#: These variants test whether an explicit abstention affordance changes
#: behavior. Forced choice still includes the abstention candidate when the
#: prompt itself does not mention it.
PROMPT_VARIANTS: dict[str, str] = {
    "no_affordance": (
        "Look at the image and answer the question in a few words."
    ),
    "standard": (
        "Look at the image and answer the question in a few words.\n"
        f"If the image does not show enough information to answer, "
        f"reply with exactly: {ABSTAIN_TEXT}"
    ),
    "last_resort": (
        "Look at the image and answer the question in a few words.\n"
        "Answer if you possibly can. Only if the image genuinely does not show "
        f"what the question asks about, reply with exactly: {ABSTAIN_TEXT}"
    ),
    #: This variant permits a short explanation without prescribing its content.
    "explain": (
        "Look at the image and answer the question in a few words.\n"
        "If the image does not show enough information to answer, say so and "
        "explain briefly why."
    ),
}

DEFAULT_VARIANT = "standard"

MAX_NEW_TOKENS = 48


@dataclass(frozen=True)
class Candidate:
    key: str  # one of CANDIDATE_KEYS
    text: str

    @property
    def is_abstain(self) -> bool:
        return self.key == "abstain"


@dataclass(frozen=True)
class Prompt:
    messages: list[dict]
    candidates: tuple[Candidate, ...] | None  # forced_choice only


def _messages(image_path: str, question: str, variant: str = DEFAULT_VARIANT) -> list[dict]:
    """Build an independent single-turn message for one image condition."""
    try:
        instruction = PROMPT_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"unknown prompt variant {variant!r}; expected one of {sorted(PROMPT_VARIANTS)}"
        ) from None
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"{instruction}\n\nQuestion: {question}"},
            ],
        }
    ]


def candidates_for(item: Item) -> tuple[Candidate, ...]:
    """The 4-way candidate set. Deduplicated by text: `options` occasionally
    repeats a string across provenances, and scoring the same continuation twice
    would double its chance of winning the argmax."""
    seen: dict[str, Candidate] = {}
    for key in ("orig", "baseline", "random"):
        text = item.options.get(key)
        if text and text not in seen:
            seen[text] = Candidate(key=key, text=text)
    seen.setdefault(ABSTAIN_TEXT, Candidate(key="abstain", text=ABSTAIN_TEXT))
    return tuple(seen.values())


def build(
    item: Item,
    condition: Condition,
    protocol: Protocol,
    image_path: str,
    variant: str = DEFAULT_VARIANT,
) -> Prompt:
    """Build the prompt for one (item, condition) under one protocol."""
    messages = _messages(image_path, item.question, variant)
    if protocol == "forced_choice":
        return Prompt(messages=messages, candidates=candidates_for(item))
    if protocol == "freeform":
        return Prompt(messages=messages, candidates=None)
    raise ValueError(f"unknown protocol: {protocol}")
