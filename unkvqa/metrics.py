"""
Grounded Abstention Informedness (GAI) and companions.

THE MEASURE
  For quant level q, restricted to items where the model DEMONSTRABLY had the
  capability (it answered the unmasked original correctly):

      GA[q]  = P(abstains on masked | i in K*)   true-positive rate
      FA[q]  = P(abstains on masked | i in A*)   false-positive rate
      GAI[q] = GA[q] - FA[q]

  This is Youden's J / Informedness (Youden 1950; Powers 2011) computed on a
  capability-conditioned subset. Using a standard statistic rather than an
  invented one is deliberate: its properties are known and it is citable. The
  contribution is the CONDITIONING, not the arithmetic.

  K* = unanswerable items answered correctly on the original by EVERY level
  A* = answerable   items answered correctly on the original by EVERY level
  B* = unanswerable items answered WRONGLY on the original by every level (diagnostic)

  Why A* is the right reference rather than B*: A* items carry the same gray
  mask but their answer is still recoverable, so abstaining there is simply
  wrong. That makes FA a clean mask-artifact control. B* is contaminated: the
  evidence really is gone there too, so a genuinely grounded model would also
  abstain, which is why B* is reported as a diagnostic and kept out of GAI.

WHY COMMON SUPPORT
  Conditioning per-level yields a different (and easier) item set at int4, so
  cross-level comparison would silently compare different populations. Every
  set here is intersected across all levels being compared.

WHY THE CONTRAST WITH J_raw CARRIES THE PAPER
  J_raw applies the same TPR-FPR arithmetic with NO capability conditioning :
  it is what anyone would report today. The expected result is accuracy flat,
  J_raw roughly flat, GAI collapsing. Reporting GAI alone invites "would an
  existing metric have caught this?" with no answer.

Calibration-family metrics (Phi_c, E-AURC, ECE) need per-candidate confidences
and are built in a later pass directly off the JSONL; they are not here.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .score import ABSTAIN, CORRECT, UNPARSEABLE, WRONG, Scored

Index = dict[tuple[str, str, str, str], Scored]  # (item, quant, protocol, condition)

BOOTSTRAP_N = 10_000
SEED = 0


# ------------------------------------------------------------------ helpers


def _outcome(idx: Index, item: str, quant: str, proto: str, cond: str) -> str | None:
    s = idx.get((item, quant, proto, cond))
    return s.outcome if s else None


def items_in(scored: Iterable[Scored], protocol: str) -> list[str]:
    return sorted({s.item_id for s in scored if s.protocol == protocol})


def _rate(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


# ------------------------------------------------------------------ sets


@dataclass
class Support:
    """The conditioning sets, intersected across all compared levels."""

    K: list[str] = field(default_factory=list)  # unanswerable, capability established
    A: list[str] = field(default_factory=list)  # answerable,   capability established
    B: list[str] = field(default_factory=list)  # unanswerable, wrong on original
    all_unanswerable: list[str] = field(default_factory=list)
    all_answerable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "K_star": len(self.K), "A_star": len(self.A), "B_star": len(self.B),
            "all_unanswerable": len(self.all_unanswerable),
            "all_answerable": len(self.all_answerable),
        }


def build_support(scored: list[Scored], protocol: str, quants: Sequence[str]) -> Support:
    idx = {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}
    meta = {s.item_id: s for s in scored if s.protocol == protocol}
    sup = Support()

    for item in items_in(scored, protocol):
        answerable = meta[item].answerable
        orig = [_outcome(idx, item, q, protocol, "original") for q in quants]
        if any(o is None for o in orig):
            continue  # incomplete across levels: cannot use for a paired comparison
        (sup.all_answerable if answerable else sup.all_unanswerable).append(item)

        if all(o == CORRECT for o in orig):
            (sup.A if answerable else sup.K).append(item)
        elif not answerable and all(o == WRONG for o in orig):
            sup.B.append(item)
    return sup


# ------------------------------------------------------------------ metrics


def _abstained(idx: Index, items: Sequence[str], quant: str, proto: str) -> list[bool]:
    return [_outcome(idx, i, quant, proto, "masked") == ABSTAIN for i in items]


@dataclass
class LevelMetrics:
    quant: str
    GA: float          # TPR on K*
    FA: float          # FPR on A*
    GAI: float         # GA - FA  <- headline
    J_raw: float       # same arithmetic, no capability conditioning
    BA: float          # diagnostic: abstention on B*
    orig_accuracy: float
    masked_answerable_accuracy: float
    abstain_rate_masked: float
    unparseable_rate: float
    pre_refusal_rate: float

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in self.__dict__.items()}


def level_metrics(scored: list[Scored], protocol: str, quant: str, sup: Support) -> LevelMetrics:
    idx = {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}
    recs = [s for s in scored if s.protocol == protocol and s.quant == quant]

    GA = _rate(_abstained(idx, sup.K, quant, protocol))
    FA = _rate(_abstained(idx, sup.A, quant, protocol))
    tpr_raw = _rate(_abstained(idx, sup.all_unanswerable, quant, protocol))
    fpr_raw = _rate(_abstained(idx, sup.all_answerable, quant, protocol))

    masked = [s for s in recs if s.condition == "masked"]
    originals = [s for s in recs if s.condition == "original"]
    masked_ans = [s for s in masked if s.answerable]

    return LevelMetrics(
        quant=quant,
        GA=GA, FA=FA, GAI=GA - FA,
        J_raw=tpr_raw - fpr_raw,
        BA=_rate(_abstained(idx, sup.B, quant, protocol)),
        orig_accuracy=_rate([s.outcome == CORRECT for s in originals]),
        masked_answerable_accuracy=_rate([s.outcome == CORRECT for s in masked_ans]),
        abstain_rate_masked=_rate([s.outcome == ABSTAIN for s in masked]),
        unparseable_rate=_rate([s.outcome == UNPARSEABLE for s in recs]),
        pre_refusal_rate=_rate([s.outcome == ABSTAIN for s in originals]),
    )


def gai_from_sets(k_flags: Sequence[bool], a_flags: Sequence[bool]) -> float:
    return _rate(k_flags) - _rate(a_flags)


# ------------------------------------------------------------------ statistics


def bootstrap_ci(
    k_flags: Sequence[bool],
    a_flags: Sequence[bool],
    stat: Callable[[Sequence[bool], Sequence[bool]], float] = gai_from_sets,
    n: int = BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Resamples K* and A* independently, because GAI is
    a difference of rates over two disjoint item sets."""
    if not k_flags or not a_flags:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    nk, na = len(k_flags), len(a_flags)
    draws = []
    for _ in range(n):
        kk = [k_flags[rng.randrange(nk)] for _ in range(nk)]
        aa = [a_flags[rng.randrange(na)] for _ in range(na)]
        draws.append(stat(kk, aa))
    draws.sort()
    lo = draws[int((alpha / 2) * n)]
    hi = draws[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    b = # items where condition 1 abstained and 2 did not; c = the reverse.
    Concordant pairs carry no information and are correctly ignored: which is
    exactly why this beats a two-proportion test here: the pairing is the design.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_abstention_test(
    scored: list[Scored], protocol: str, q1: str, q2: str, items: Sequence[str]
) -> dict:
    idx = {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}
    b = c = 0
    for i in items:
        a1 = _outcome(idx, i, q1, protocol, "masked") == ABSTAIN
        a2 = _outcome(idx, i, q2, protocol, "masked") == ABSTAIN
        if a1 and not a2:
            b += 1
        elif a2 and not a1:
            c += 1
    return {"pair": f"{q1}_vs_{q2}", "b": b, "c": c, "n_discordant": b + c,
            "p_exact": mcnemar_exact(b, c)}


def paired_gai_delta(
    scored: list[Scored], protocol: str, q1: str, q2: str, sup: Support,
    n: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = SEED,
) -> dict:
    """Bootstrap CI for GAI[q1] - GAI[q2], resampling items PAIRED across levels.

    This is the correct test for "did GAI decline", and it is much more powerful
    than eyeballing whether the two marginal CIs overlap: a comparison that is
    statistically invalid and, here, badly underpowered. The same items are
    measured at both levels, so item difficulty is shared and cancels: each
    bootstrap draw picks item indices ONCE and evaluates both levels on that same
    draw, keeping the pairing intact.
    """
    idx = {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}

    def flags(items, quant):
        return [_outcome(idx, i, quant, protocol, "masked") == ABSTAIN for i in items]

    k1, k2 = flags(sup.K, q1), flags(sup.K, q2)
    a1, a2 = flags(sup.A, q1), flags(sup.A, q2)
    if not sup.K or not sup.A:
        return {"pair": f"{q1}_minus_{q2}", "delta": float("nan"),
                "ci95": [float("nan"), float("nan")], "p_two_sided": float("nan")}

    point = (_rate(k1) - _rate(a1)) - (_rate(k2) - _rate(a2))
    rng = random.Random(seed)
    nk, na = len(sup.K), len(sup.A)
    draws = []
    for _ in range(n):
        ki = [rng.randrange(nk) for _ in range(nk)]
        ai = [rng.randrange(na) for _ in range(na)]
        g1 = _rate([k1[j] for j in ki]) - _rate([a1[j] for j in ai])
        g2 = _rate([k2[j] for j in ki]) - _rate([a2[j] for j in ai])
        draws.append(g1 - g2)
    draws.sort()
    lo = draws[int((alpha / 2) * n)]
    hi = draws[min(n - 1, int((1 - alpha / 2) * n))]
    # Two-sided bootstrap p: how much mass sits on the wrong side of zero.
    below = sum(1 for d in draws if d <= 0)
    p = 2.0 * min(below, n - below) / n
    return {"pair": f"{q1}_minus_{q2}", "delta": point,
            "ci95": [lo, hi], "p_two_sided": min(1.0, p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction, with monotonicity enforced."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (key, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - rank) * p))
        out[key] = adj
        prev = adj
    return out


def flip_rate(scored: list[Scored], protocol: str, q1: str, q2: str,
              both_correct_only: bool = True) -> float:
    """Fraction of (item, condition) pairs whose chosen answer differs between
    two levels. Restricted by default to cases BOTH levels got right: that is
    the version that makes the point, because it shows behaviour changing while
    accuracy does not."""
    idx = {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}
    keys = {(s.item_id, s.condition) for s in scored if s.protocol == protocol}
    flags = []
    for item, cond in sorted(keys):
        s1, s2 = idx.get((item, q1, protocol, cond)), idx.get((item, q2, protocol, cond))
        if not s1 or not s2:
            continue
        if both_correct_only and not (s1.outcome == CORRECT and s2.outcome == CORRECT):
            continue
        flags.append(s1.chosen.strip().casefold() != s2.chosen.strip().casefold())
    return _rate(flags)


# ------------------------------------------------------------------ top level


def evaluate(scored: list[Scored], protocol: str, quants: Sequence[str]) -> dict:
    """Full metric bundle for one protocol across the given levels."""
    idx = {(s.item_id, s.quant, s.protocol, s.condition): s for s in scored}
    sup = build_support(scored, protocol, quants)

    levels, cis = {}, {}
    for q in quants:
        levels[q] = level_metrics(scored, protocol, q, sup).as_dict()
        cis[q] = bootstrap_ci(
            _abstained(idx, sup.K, q, protocol), _abstained(idx, sup.A, q, protocol)
        )

    pairs = [(quants[i], quants[j])
             for i in range(len(quants)) for j in range(i + 1, len(quants))]
    tests = {}
    for q1, q2 in pairs:
        t = paired_abstention_test(scored, protocol, q1, q2, sup.K)
        # The same test on A*, where MORE abstention is WRONG. This is where the
        # action is: quantization moves FA, not GA, so testing only K* would miss
        # the effect entirely.
        ta = paired_abstention_test(scored, protocol, q1, q2, sup.A)
        t["A_b"], t["A_c"] = ta["b"], ta["c"]
        t["A_p_exact"] = ta["p_exact"]
        t["flip_rate_both_correct"] = flip_rate(scored, protocol, q1, q2, True)
        tests[t["pair"]] = t
    adjusted = holm({k: v["p_exact"] for k, v in tests.items()})
    adjusted_a = holm({k: v["A_p_exact"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm"] = adjusted[k]
        v["A_p_holm"] = adjusted_a[k]

    deltas = {}
    for q1, q2 in pairs:
        d = paired_gai_delta(scored, protocol, q1, q2, sup)
        deltas[d["pair"]] = d

    return {
        "protocol": protocol,
        "support": sup.as_dict(),
        "levels": levels,
        "gai_ci95": {q: list(cis[q]) for q in quants},
        "pairwise": tests,
        "gai_deltas": deltas,
    }
