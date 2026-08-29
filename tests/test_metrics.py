"""
Synthetic tests for the scorer and GAI.

These run BEFORE any real numbers are trusted. A metric that cannot recover the
right answer on data whose ground truth is known by construction is wrong no
matter how plausible its output looks on the real run. The degenerate-strategy
cases are the important ones: abstain-always, abstain-never, and abstain-at-
random must all score GAI = 0, because a metric that rewards indiscriminate
refusal would make the whole paper's claim unfalsifiable.

Run:  python tests/test_metrics.py     (also works under pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from unkvqa.metrics import (
    build_support, evaluate, flip_rate, holm, level_metrics, mcnemar_exact,
)
from unkvqa.score import (
    ABSTAIN, CORRECT, UNPARSEABLE, WRONG, Scored, answer_matches, decide_forced,
    is_abstention, normalize, score_record, soft_normalize,
)

QUANTS = ["bf16", "int8", "int4"]
PROTO = "forced_choice"


# ------------------------------------------------------------- synthetic data


def make(item_id, quant, condition, answerable, outcome, chosen="x"):
    return Scored(
        item_id=item_id, quant=quant, protocol=PROTO, variant="standard",
        condition=condition, answerable=answerable, reason=None,
        should_abstain=(condition == "masked" and not answerable),
        outcome=outcome, chosen=chosen, raw_output=None,
    )


def synth(n_unans=100, n_ans=100, masked_policy=None, orig_correct=True, quants=QUANTS):
    """Build a scored set where every item is CORRECT on the original (so K*/A*
    are the full sets), and the masked outcome follows `masked_policy`."""
    out = []
    for q in quants:
        for k in range(n_unans):
            i = f"u{k}"
            out.append(make(i, q, "original", False,
                            CORRECT if orig_correct else WRONG))
            out.append(make(i, q, "masked", False, masked_policy(i, q, False)))
        for k in range(n_ans):
            i = f"a{k}"
            out.append(make(i, q, "original", True,
                            CORRECT if orig_correct else WRONG))
            out.append(make(i, q, "masked", True, masked_policy(i, q, True)))
    return out


def gai_of(scored, quant, quants=QUANTS):
    sup = build_support(scored, PROTO, quants)
    return level_metrics(scored, PROTO, quant, sup).GAI


# ------------------------------------------------------------- degenerate cases


def test_abstain_always_is_zero():
    s = synth(masked_policy=lambda i, q, ans: ABSTAIN)
    for q in QUANTS:
        assert abs(gai_of(s, q)) < 1e-9, f"abstain-always must be 0, got {gai_of(s, q)}"


def test_abstain_never_is_zero():
    s = synth(masked_policy=lambda i, q, ans: WRONG)
    for q in QUANTS:
        assert abs(gai_of(s, q)) < 1e-9, f"abstain-never must be 0, got {gai_of(s, q)}"


def test_random_abstention_is_about_zero():
    rng = random.Random(7)
    s = synth(n_unans=3000, n_ans=3000,
              masked_policy=lambda i, q, ans: ABSTAIN if rng.random() < 0.6 else WRONG)
    for q in QUANTS:
        assert abs(gai_of(s, q)) < 0.05, f"random-at-rate must be ~0, got {gai_of(s, q)}"


def test_perfect_discriminator_is_one():
    # Abstains exactly when it should: on masked unanswerable, never on answerable.
    s = synth(masked_policy=lambda i, q, ans: WRONG if ans else ABSTAIN)
    for q in QUANTS:
        assert abs(gai_of(s, q) - 1.0) < 1e-9, f"perfect must be 1, got {gai_of(s, q)}"


def test_inverted_discriminator_is_negative_one():
    s = synth(masked_policy=lambda i, q, ans: ABSTAIN if ans else WRONG)
    for q in QUANTS:
        assert abs(gai_of(s, q) + 1.0) < 1e-9


def test_gai_invariant_to_uniform_abstention_shift():
    """The property that earns GAI its place: adding a uniform tendency to
    abstain must not change it. Here 40% of BOTH sets flip to abstaining."""
    rng = random.Random(11)
    base = {}

    def policy(i, q, ans):
        key = (i, q)
        if key not in base:
            base[key] = WRONG if ans else ABSTAIN  # perfect discriminator
        if rng.random() < 0.4:
            return ABSTAIN
        return base[key]

    s = synth(n_unans=4000, n_ans=4000, masked_policy=policy)
    for q in QUANTS:
        g = gai_of(s, q)
        assert 0.4 < g < 0.8, f"expected a damped-but-positive GAI, got {g}"


# ------------------------------------------------------------- conditioning


def test_common_support_excludes_items_any_level_gets_wrong():
    s = []
    for q in QUANTS:
        # u0 correct everywhere; u1 wrong at int4 only -> excluded from K*
        s.append(make("u0", q, "original", False, CORRECT))
        s.append(make("u0", q, "masked", False, ABSTAIN))
        s.append(make("u1", q, "original", False, WRONG if q == "int4" else CORRECT))
        s.append(make("u1", q, "masked", False, ABSTAIN))
        s.append(make("a0", q, "original", True, CORRECT))
        s.append(make("a0", q, "masked", True, WRONG))
    sup = build_support(s, PROTO, QUANTS)
    assert sup.K == ["u0"], f"K* should exclude u1, got {sup.K}"
    assert sup.A == ["a0"]


def test_pre_refusal_items_excluded_from_all_sets():
    """A model that already refused the ORIGINAL tells us nothing about evidence
    sensitivity, so such items must land in neither K* nor B*."""
    s = []
    for q in QUANTS:
        s.append(make("u0", q, "original", False, ABSTAIN))
        s.append(make("u0", q, "masked", False, ABSTAIN))
    sup = build_support(s, PROTO, QUANTS)
    assert sup.K == [] and sup.B == []
    m = level_metrics(s, PROTO, "bf16", sup)
    assert abs(m.pre_refusal_rate - 1.0) < 1e-9


def test_gai_differs_from_j_raw_when_capability_varies():
    """The contrast the paper rests on: J_raw and GAI must be able to diverge."""
    s = []
    for q in QUANTS:
        for k in range(50):  # capable + grounded
            s.append(make(f"u{k}", q, "original", False, CORRECT))
            s.append(make(f"u{k}", q, "masked", False, ABSTAIN))
        for k in range(50, 100):  # NOT capable, still abstains -> inflates J_raw
            s.append(make(f"u{k}", q, "original", False, WRONG))
            s.append(make(f"u{k}", q, "masked", False, ABSTAIN))
        for k in range(50):
            s.append(make(f"a{k}", q, "original", True, CORRECT))
            s.append(make(f"a{k}", q, "masked", True, ABSTAIN if k < 25 else WRONG))
    sup = build_support(s, PROTO, QUANTS)
    m = level_metrics(s, PROTO, "bf16", sup)
    assert len(sup.K) == 50 and len(sup.B) == 50
    assert abs(m.J_raw - 0.5) < 1e-9, m.J_raw
    assert abs(m.GAI - 0.5) < 1e-9, m.GAI
    assert abs(m.BA - 1.0) < 1e-9  # B* diagnostic picks up the ungrounded half


# ------------------------------------------------------------- statistics


def test_mcnemar_symmetry_and_extremes():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-9
    assert mcnemar_exact(20, 0) < 1e-5
    assert mcnemar_exact(10, 2) == mcnemar_exact(2, 10)  # symmetric in b,c


def test_holm_is_monotone_and_conservative():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] >= 0.01 and adj["b"] >= 0.04
    assert adj["a"] <= adj["c"] <= adj["b"]  # ordering preserved


def test_flip_rate_counts_only_agreed_correct_by_default():
    s = []
    for q in QUANTS:
        s.append(make("a0", q, "original", True, CORRECT, chosen="dog"))
        s.append(make("a0", q, "masked", True, CORRECT,
                      chosen="cat" if q == "int4" else "dog"))
    assert flip_rate(s, PROTO, "bf16", "int8") == 0.0
    assert flip_rate(s, PROTO, "bf16", "int4") == 0.5  # 1 of 2 conditions flipped


# ------------------------------------------------------------- scorer


def test_normalize_and_matching():
    assert normalize("The Dog's!") == "dog s"
    assert normalize("Two") == "2"
    assert answer_matches("a dog", "dog")
    assert answer_matches("stop sign", "sign")           # lenient subset
    assert not answer_matches("stop sign", "sign", lenient=False)
    assert not answer_matches("tissue", "toilet paper")  # honest limitation
    assert not answer_matches("", "dog")


def test_abstention_detection():
    for t in ["unanswerable", "Unanswerable.", "I don't know", "I cannot answer",
              "It's impossible to tell", "not enough information",
              "The object is not visible",
              # free-form refusals seen under the `explain` variant, where the
              # model is not told which string to emit
              "The image does not provide enough information to determine",
              "The image does not show the object clearly",
              "There is not sufficient detail to answer"]:
        assert is_abstention(t), t
    for t in ["a dog", "skiing", "phone", "two men playing frisbee"]:
        assert not is_abstention(t), t


def test_contractions_survive_normalisation():
    """Regression guard. `normalize` mangles "don't" into "don t", which silently
    broke every apostrophe-bearing refusal pattern. Models also emit the Unicode
    right-single-quote, so that must fold to ASCII too."""
    assert soft_normalize("I don't know") == "i don't know"
    assert soft_normalize("I don’t know") == "i don't know"
    for variant in ["I don't know", "I don’t know", "i dont know",
                    "I do not know", "I DON'T KNOW!"]:
        assert is_abstention(variant), variant


def test_forced_choice_rule_matters():
    """Regression guard for the surface-form-competition finding: mean and sum
    disagree, and sum is the rule that matched the model's own freeform output."""
    lp = {
        "abstain": {"text": "unanswerable", "logprob_sum": -1.375,
                    "logprob_mean": -0.458, "n_tokens": 3},
        "orig": {"text": "phone", "logprob_sum": -0.500,
                 "logprob_mean": -0.500, "n_tokens": 1},
    }
    assert decide_forced(lp, "logprob_sum")[0] == "orig"
    assert decide_forced(lp, "logprob_mean")[0] == "abstain"


def test_answering_an_unanswerable_item_is_never_correct():
    rec = {
        "item_id": "v:1", "quant": "int4", "protocol": "freeform",
        "condition": "masked", "answerable": False, "reason": "3",
        "should_abstain": True, "gold_orig": "phone", "gold_masked": None,
        "raw_output": "phone",
    }
    scored = score_record(rec)
    assert scored.outcome == WRONG
    # The task label is wrong by construction after removal, but we retain the
    # diagnostic that the model repeated the now-unsupported source answer.
    assert scored.reference_match is True


def test_unrelated_wrong_answer_is_not_reference_leakage():
    rec = {
        "item_id": "v:2", "quant": "int4", "protocol": "freeform",
        "condition": "masked", "answerable": False, "reason": "3",
        "should_abstain": True, "gold_orig": "phone", "gold_masked": None,
        "raw_output": "sandwich",
    }
    scored = score_record(rec)
    assert scored.outcome == WRONG
    assert scored.reference_match is False


def test_absence_reports_are_not_hallucinations():
    """"nothing" / "no one" on an image whose object was removed is a correct
    observation, not a wrong answer. Scoring it WRONG understated GA badly in
    the no_affordance condition."""
    from unkvqa.score import ABSENCE, is_absence_report

    base = {
        "item_id": "p:1", "quant": "bf16", "protocol": "freeform",
        "condition": "target", "should_abstain": True, "gold": None,
        "answerable": False, "reason": None,
        "gold_orig": "sandwich", "gold_masked": None,
    }
    for text in ["nothing", "no one", "neither", "none",
                 "There is no sandwich in the image",
                 # bare "no <noun>": a smoke test scored "no animal" as a wrong
                 # answer when the horse had been removed
                 "no animal", "no food", "no animal visible"]:
        assert score_record({**base, "raw_output": text}).outcome == ABSENCE, text
    # a real hallucination stays WRONG
    assert score_record({**base, "raw_output": "pizza"}).outcome == WRONG
    # explicit refusal stays ABSTAIN, not ABSENCE
    assert score_record({**base, "raw_output": "unanswerable"}).outcome == ABSTAIN


def test_nothing_as_a_legitimate_gold_is_correct_not_absence():
    """"What is on the plate?" -> "nothing" is an ordinary VQA answer. The gold
    guard must stop it being read as a report of absence."""
    rec = {
        "item_id": "p:2", "quant": "bf16", "protocol": "freeform",
        "condition": "control", "should_abstain": False, "gold": "nothing",
        "answerable": True, "reason": None,
        "gold_orig": "nothing", "gold_masked": "nothing",
        "raw_output": "nothing",
    }
    assert score_record(rec).outcome == CORRECT
    from unkvqa.score import is_absence_report
    assert not is_absence_report("nothing", gold="nothing")
    assert is_absence_report("nothing", gold="sandwich")


def test_empty_freeform_is_unparseable_not_answered():
    rec = {
        "item_id": "v:1", "quant": "int4", "protocol": "freeform",
        "condition": "masked", "answerable": True, "reason": None,
        "should_abstain": False, "gold_orig": "a", "gold_masked": "b",
        "raw_output": "   ",
    }
    assert score_record(rec).outcome == UNPARSEABLE


def test_evaluate_end_to_end_shape():
    s = synth(n_unans=40, n_ans=40,
              masked_policy=lambda i, q, ans: WRONG if ans else ABSTAIN)
    out = evaluate(s, PROTO, QUANTS)
    assert out["support"]["K_star"] == 40 and out["support"]["A_star"] == 40
    assert abs(out["levels"]["int4"]["GAI"] - 1.0) < 1e-9
    lo, hi = out["gai_ci95"]["int4"]
    assert lo <= 1.0 <= hi + 1e-9
    assert set(out["pairwise"]) == {"bf16_vs_int8", "bf16_vs_int4", "int8_vs_int4"}


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
