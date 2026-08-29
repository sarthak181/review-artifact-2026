"""Convert saved generation records into evaluation outcomes.

Correct, wrong, abstaining, absence-reporting, and unparseable responses remain
distinct. Forced-choice evaluation defaults to summed candidate-token log
probability; the mean rule remains available for sensitivity checks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Literal

Outcome = Literal["CORRECT", "WRONG", "ABSTAIN", "ABSENCE", "UNPARSEABLE"]
Rule = Literal["logprob_sum", "logprob_mean"]

CORRECT: Outcome = "CORRECT"
WRONG: Outcome = "WRONG"
ABSTAIN: Outcome = "ABSTAIN"
#: Asserts the thing is not there ("nothing", "no one") rather than declining to
#: answer. Grounded, but a different behaviour from refusal: see is_absence_report.
ABSENCE: Outcome = "ABSENCE"
UNPARSEABLE: Outcome = "UNPARSEABLE"

#: Both count as evidence the model registered the removal.
GROUNDED_OUTCOMES = (ABSTAIN, ABSENCE)

# ---------------------------------------------------------------- normalisation

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_PUNCT_KEEP_APOS = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")
_UNICODE_APOS = {"’": "'", "ʼ": "'", "´": "'"}
_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize(text: str | None) -> str:
    """VQA-style answer normalisation: casefold, strip punctuation and articles,
    map number words to digits, collapse whitespace.

    Used for ANSWER MATCHING only. It destroys contractions ("don't" -> "don t"),
    which is harmless for comparing short answers but fatal for phrase matching :
    hence `soft_normalize` below.
    """
    if not text:
        return ""
    t = _PUNCT.sub(" ", text.casefold())
    tokens = [_NUMBERS.get(w, w) for w in _WS.split(t) if w and w not in _ARTICLES]
    return " ".join(tokens)


def soft_normalize(text: str | None) -> str:
    """Casefold and strip punctuation but KEEP apostrophes, so contractions
    survive intact.

    This exists because of a bug the synthetic tests caught: `normalize` turns
    "I don't know" into "i don t know", so every apostrophe-bearing refusal
    pattern silently failed to match and those abstentions would have been
    scored as answers. Unicode apostrophes are folded to ASCII first, since
    models emit U+2019 freely.
    """
    if not text:
        return ""
    t = text.casefold()
    for uni, ascii_ in _UNICODE_APOS.items():
        t = t.replace(uni, ascii_)
    return _WS.sub(" ", _PUNCT_KEEP_APOS.sub(" ", t)).strip()


# ---------------------------------------------------------------- abstention

#: Ordered most-specific first. Matched against the NORMALISED output.
_ABSTAIN_PATTERNS = [
    r"^unanswerable$",
    r"\bunanswerable\b",
    r"\bcannot be (answered|determined)\b",
    r"\bcan(no|')?t be (answered|determined)\b",
    r"\bi (do not|don'?t) know\b",
    r"\bi (cannot|can'?t|am unable to) (answer|tell|determine|see)\b",
    # Allow words between the negation and "enough information". A smoke test
    # scored "The image does not PROVIDE enough information..." as an answer
    # because the original pattern required the two to be adjacent.
    r"\bnot( \w+){0,3} (enough|sufficient) (information|detail|data|context)\b",
    r"\b(does|do|did) not (show|contain|provide|include|display)\b",
    r"\b(there is|there'?s) not enough\b",
    r"\bimpossible to (tell|determine|answer)\b",
    r"\bunable to (answer|determine|tell)\b",
    r"\bnot (visible|shown|possible to tell)\b",
    r"\bno(t)? (clear|discernible)\b",
    r"\bi('?m| am) not sure\b",
]
_ABSTAIN_RE = [re.compile(p) for p in _ABSTAIN_PATTERNS]


def is_abstention(text: str | None) -> bool:
    """Matched against `soft_normalize`, NOT `normalize`: contractions must
    survive or half these patterns never fire."""
    norm = soft_normalize(text)
    return any(rx.search(norm) for rx in _ABSTAIN_RE)


#: ABSENCE REPORTS: the model names the absence instead of refusing.
#:
#: Discovered in the no_affordance condition, where the model answered "nothing",
#: "no one", "neither" on images whose answer object had been removed. Those are
#: CORRECT observations about the scene and evidence of grounding: scoring them
#: as hallucinations understates the model badly (it put GA at 0.024 when the
#: model was in fact detecting the removal ~20% of the time).
#:
#: Kept SEPARATE from abstention rather than merged into it, because the two are
#: different behaviours: "unanswerable" declines the question, "nothing" answers
#: it. Which of them should count toward grounding is a reporting decision, so
#: the scorer records the distinction and lets the metric layer choose.
_ABSENCE_PATTERNS = [
    r"^nothing$", r"^none$", r"^no one$", r"^nobody$", r"^neither$",
    r"^nothing (is |at all)?\w*$",
    # Bare "no <noun>": caught by a smoke test where "no animal" (horse removed)
    # scored as a wrong answer rather than as a report of absence.
    r"^no \w+$",
    r"^no \w+ (visible|shown|present|there|here)$",
    r"\bthere (is|are) (no|nothing|none)\b",
    r"\bno \w+ (is |are )?(visible|shown|present|there|in the (image|picture|photo))\b",
    r"\b(is|are) not (visible|shown|present|there)\b",
    r"\b(he|she|they|it) is not \w+ing\b",
]
_ABSENCE_RE = [re.compile(p) for p in _ABSENCE_PATTERNS]


def is_absence_report(text: str | None, gold: str | None = None) -> bool:
    """True when the output asserts the thing is not there.

    `gold` guards the ambiguous case: "nothing" is a perfectly ordinary VQA
    answer ("What is on the plate?" -> "nothing"), so when it MATCHES the gold it
    is a correct answer, not an absence report.
    """
    if not text:
        return False
    if gold and answer_matches(text, gold):
        return False
    norm = soft_normalize(text)
    return any(rx.search(norm) for rx in _ABSENCE_RE)


# ---------------------------------------------------------------- matching


def answer_matches(pred: str | None, gold: str | None, lenient: bool = True) -> bool:
    """Strict = normalised equality. Lenient additionally accepts whole-word
    subset containment in either direction ("stop sign" vs "sign").

    Deliberately does NOT attempt synonymy: 'tissue' for gold 'toilet paper' is
    a real case in the data and no string rule catches it. Those land in WRONG
    and are recovered, if at all, by the deferred judge pass. The strict/lenient
    gap is reported so the size of that grey zone is visible.
    """
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if not lenient:
        return False
    pt, gt = set(p.split()), set(g.split())
    return bool(pt) and bool(gt) and (pt <= gt or gt <= pt)


# ---------------------------------------------------------------- decisions


def decide_forced(option_logprobs: dict, rule: Rule = "logprob_sum") -> tuple[str, str]:
    """Return (winning_key, winning_text) under the given decision rule."""
    if not option_logprobs:
        return "", ""
    key = max(option_logprobs, key=lambda k: option_logprobs[k][rule])
    return key, option_logprobs[key]["text"]


@dataclass(frozen=True)
class Scored:
    item_id: str
    quant: str
    protocol: str
    variant: str
    condition: str
    answerable: bool
    reason: str | None
    should_abstain: bool
    outcome: Outcome
    chosen: str  # the answer the model effectively gave ("" if abstained)
    raw_output: str | None
    # Whether the emitted content matches the pre-intervention reference answer.
    # This is deliberately separate from `outcome`: on an evidence-removed item,
    # emitting the old answer is a confabulation (WRONG) but is still important
    # diagnostic evidence of language-prior leakage.
    reference_match: bool = False


def score_record(rec: dict, rule: Rule = "logprob_sum", lenient: bool = True) -> Scored:
    """Score one JSONL record into an outcome."""
    condition = rec["condition"]
    if "gold" in rec:
        # Written per-condition by the runner. Authoritative: it removes any
        # chance of the scorer picking the wrong gold field for a condition,
        # which matters because original and masked golds disagree ~64% of the
        # time on UNK-VQA and the probe has three conditions rather than two.
        gold = rec["gold"]
    else:
        gold = rec["gold_orig"] if condition == "original" else rec.get("gold_masked")

    if rec["protocol"] == "forced_choice":
        lp = rec.get("option_logprobs") or {}
        key, text = decide_forced(lp, rule)
        if not key:
            outcome, chosen = UNPARSEABLE, ""
        elif key == "abstain":
            outcome, chosen = ABSTAIN, ""
        else:
            chosen = text
            outcome = CORRECT if answer_matches(text, gold, lenient) else WRONG
    else:  # freeform
        raw = rec.get("raw_output")
        if raw is None or not raw.strip():
            outcome, chosen = UNPARSEABLE, ""
        elif is_abstention(raw):
            outcome, chosen = ABSTAIN, ""
        elif answer_matches(raw, gold, lenient):
            # Checked BEFORE absence, so a legitimate gold of "nothing" is
            # scored CORRECT rather than mistaken for a report of absence.
            outcome, chosen = CORRECT, raw.strip()
        elif is_absence_report(raw, gold):
            outcome, chosen = ABSENCE, raw.strip()
        else:
            outcome, chosen = WRONG, raw.strip()

    # Preserve whether the emitted content matches the pre-intervention answer,
    # independently of the per-condition task label. On a removed-evidence item
    # the condition gold is intentionally null, so `outcome == CORRECT` cannot
    # recover this diagnostic.
    # Probe records retain the source answer as `answer`; UNK-VQA records use
    # `gold_orig`.  Per-condition `gold` is null at complete target removal.
    source_gold = rec.get("gold_orig", rec.get("answer", rec.get("gold")))
    reference_match = bool(chosen) and answer_matches(chosen, source_gold, lenient)

    # An unanswerable masked item has no correct answer, so "CORRECT" is not
    # reachable there by construction: answering at all is the error.
    if rec["should_abstain"] and outcome == CORRECT:
        outcome = WRONG

    return Scored(
        item_id=rec["item_id"], quant=rec["quant"], protocol=rec["protocol"],
        variant=rec.get("variant", "standard"),
        condition=condition,
        # UNK-VQA carries `answerable` per item. The constructed probe does not:
        # answerability there is a property of the CONDITION, not the item, so it
        # is derived from `should_abstain` instead.
        answerable=rec.get("answerable", not rec["should_abstain"]),
        reason=rec.get("reason"),
        should_abstain=rec["should_abstain"], outcome=outcome, chosen=chosen,
        raw_output=rec.get("raw_output"), reference_match=reference_match,
    )


def load_scored(
    path: str,
    rule: Rule = "logprob_sum",
    lenient: bool = True,
    variant: str | None = "standard",
) -> list[Scored]:
    """Load and score a JSONL.

    `variant` filters to one prompt variant: pass None to load all. Filtering
    matters: variants must never be pooled, since the whole point of running them
    is that they produce different abstention rates on the same items.
    """
    out: list[Scored] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = score_record(json.loads(line), rule, lenient)
            if variant is None or s.variant == variant:
                out.append(s)
    return out


def by_key(scored: Iterable[Scored]) -> dict[tuple[str, str, str, str], Scored]:
    """Index as (item_id, quant, protocol, condition) -> Scored."""
    return {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}
