"""Regression tests for the graded dose-response estimator."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import graded_eval as G
from unkvqa.score import ABSENCE, ABSTAIN, CORRECT, WRONG, Scored


def scored(item, condition, outcome):
    return Scored(
        item_id=item, quant="bf16", protocol="freeform", variant="standard",
        condition=condition, answerable=condition == "original", reason=None,
        should_abstain=condition != "original", outcome=outcome, chosen="",
        raw_output="",
    )


def test_equal_coverages_produce_one_point_per_nominal_level():
    """Tied x values must not become a staircase through tuple sorting."""
    idx = {}
    coverage = {}
    items = [f"i{k}" for k in range(4)]
    for k, item in enumerate(items):
        idx[(item, "bf16", "original")] = scored(item, "original", CORRECT)
        idx[(item, "bf16", "target_100")] = scored(
            item, "target_100", ABSTAIN if k < 2 else WRONG
        )
        coverage[item] = {"levels": {"100": 1.0}}

    points = G.level_curve(idx, items, "bf16", coverage)
    assert [(p.level, p.coverage, p.grounded_rate, p.n) for p in points] == [
        (0, 0.0, 0.0, 4),
        (100, 1.0, 0.5, 4),
    ]
    assert G.threshold50(points) == 1.0


def test_threshold_is_linearly_interpolated_between_level_rates():
    points = [
        G.CurvePoint(0, 0.0, 0.1, 10),
        G.CurvePoint(50, 0.4, 0.3, 10),
        G.CurvePoint(100, 0.8, 0.7, 10),
    ]
    assert abs(G.threshold50(points) - 0.6) < 1e-12


# --------------------------------------------------------------------------
# Degenerate-strategy validity for the PAPER'S estimator.
#
# test_metrics.py proves always/never/random abstention score GAI = 0, but it
# does so against unkvqa.metrics.level_metrics, whose grounded set is ABSTAIN
# only. The headline numbers in the paper come from graded_eval instead, whose
# grounded set is ABSTAIN OR ABSENCE and whose GAI/bootstrap are separate code.
# These tests pin the same guarantees onto that actual estimator, so a strategy
# that discriminates nothing cannot score above zero.
# --------------------------------------------------------------------------


def _full(item, condition, outcome, quant="bf16", reference_match=False):
    return Scored(
        item_id=item, quant=quant, protocol="freeform", variant="standard",
        condition=condition, answerable=(condition == "original"), reason=None,
        should_abstain=(condition not in ("original", "control")),
        outcome=outcome, chosen="", raw_output="", reference_match=reference_match,
    )


def _removal_index(pairs, quant="bf16"):
    """pairs: list of (target_100_outcome, control_outcome). Returns (idx, items)
    shaped exactly as graded_eval.full_removal_metrics/bootstrap_gai consume."""
    idx, items = {}, []
    for k, (t, c) in enumerate(pairs):
        i = f"i{k}"
        items.append(i)
        idx[(i, quant, "target_100")] = _full(i, "target_100", t, quant)
        idx[(i, quant, "control")] = _full(i, "control", c, quant)
    return idx, items


def test_full_removal_gai_zero_for_degenerate_strategies():
    # abstain-always: grounded on BOTH target and control -> GR = FAR = 1 -> 0.
    idx, items = _removal_index([(ABSTAIN, ABSTAIN)] * 50)
    assert G.full_removal_metrics(idx, items, "bf16")["gai"] == 0.0
    # never-abstain: answers on both -> GR = FAR = 0 -> 0.
    idx, items = _removal_index([(WRONG, WRONG)] * 50)
    assert G.full_removal_metrics(idx, items, "bf16")["gai"] == 0.0


def test_full_removal_gai_one_for_perfect_discriminator():
    idx, items = _removal_index([(ABSTAIN, WRONG)] * 50)
    m = G.full_removal_metrics(idx, items, "bf16")
    assert m["grounded_target"] == 1.0 and m["grounded_control"] == 0.0
    assert m["gai"] == 1.0


def test_absence_reports_count_as_grounded_like_abstention():
    # The paper's grounded set is ABSTAIN OR ABSENCE; an absence report on the
    # target must lift GAI exactly as an explicit refusal does.
    idx, items = _removal_index([(ABSENCE, WRONG)] * 40)
    assert G.full_removal_metrics(idx, items, "bf16")["gai"] == 1.0


def test_full_removal_gai_about_zero_for_random_abstention():
    rng = random.Random(3)
    pairs = [(ABSTAIN if rng.random() < 0.5 else WRONG,
              ABSTAIN if rng.random() < 0.5 else WRONG) for _ in range(4000)]
    idx, items = _removal_index(pairs)
    assert abs(G.full_removal_metrics(idx, items, "bf16")["gai"]) < 0.05


def test_bootstrap_gai_collapses_on_degenerate_supports():
    # Every item identical, so every resample yields the same GAI and the CI is a
    # point. Guards against a bootstrap that silently widens or mislabels bounds.
    idx, items = _removal_index([(ABSTAIN, ABSTAIN)] * 50)
    assert G.bootstrap_gai(idx, items, "bf16", n=500, seed=1) == (0.0, 0.0)
    idx, items = _removal_index([(ABSTAIN, WRONG)] * 50)
    assert G.bootstrap_gai(idx, items, "bf16", n=500, seed=1) == (1.0, 1.0)


def test_reference_leakage_tracks_the_reference_match_flag():
    # RefLeak is measured on the target condition, independent of the outcome:
    # half the items repeat the removed answer (reference_match=True).
    idx, items = _removal_index([(WRONG, WRONG)] * 10)
    for k in range(5):
        i = f"i{k}"
        idx[(i, "bf16", "target_100")] = _full(i, "target_100", WRONG, reference_match=True)
    assert abs(G.full_removal_metrics(idx, items, "bf16")["reference_leakage"] - 0.5) < 1e-9


def test_logistic_threshold_recovers_known_midpoint():
    # Per-level grounded fractions follow p(x) = 1 / (1 + exp(-10 (x - 0.7))),
    # so the logistic MLE must recover the 0.7 midpoint up to rounding.
    import math
    levels = {"11": 0.2, "23": 0.4, "37": 0.6, "54": 0.8, "73": 0.95, "100": 1.0}
    n = 2000
    idx, coverage, items = {}, {}, [f"i{k}" for k in range(n)]
    for item in items:
        coverage[item] = {"levels": dict(levels)}
        idx[(item, "bf16", "original")] = scored(item, "original", WRONG)
    for level, x in [("0", 0.0)] + list(levels.items()):
        grounded = round(n / (1 + math.exp(-10 * (x - 0.7))))
        cond = "original" if level == "0" else f"target_{level}"
        for k, item in enumerate(items):
            idx[(item, "bf16", cond)] = scored(item, cond, ABSTAIN if k < grounded else WRONG)
    value = G.logistic_threshold50(idx, items, "bf16", coverage)
    assert value is not None and abs(value - 0.7) < 0.005


def test_logistic_threshold_undefined_without_any_grounded_response():
    idx, items = {}, ["i0", "i1"]
    coverage = {i: {"levels": {"100": 1.0}} for i in items}
    for i in items:
        idx[(i, "bf16", "original")] = scored(i, "original", CORRECT)
        idx[(i, "bf16", "target_100")] = scored(i, "target_100", WRONG)
    assert G.logistic_threshold50(idx, items, "bf16", coverage) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
