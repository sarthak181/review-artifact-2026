"""Evaluate graded object removal without tie-ordering artifacts.

Each intervention level has a per-item measurement of the fraction of target
object pixels covered.  We compute one response rate per level and place it at
that level's median measured coverage.  The 50% threshold is interpolated
between adjacent levels.  Item bootstrap resampling keeps repeated levels for
an item together and yields paired uncertainty.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from dataclasses import dataclass

from unkvqa.score import ABSENCE, ABSTAIN, CORRECT, Scored, score_record

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ("bf16", "int8", "int4")
LEVELS = (0, 11, 23, 37, 54, 73, 100)
GROUNDED = (ABSTAIN, ABSENCE)


@dataclass(frozen=True)
class CurvePoint:
    level: int
    coverage: float
    grounded_rate: float
    n: int


def _rate(flags) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else float("nan")


def _quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = p * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (pos - lo) * (values[hi] - values[lo])


def condition_for(level: int) -> str:
    return "original" if level == 0 else f"target_{level}"


def load(path: str, variant: str = "standard", protocol: str = "freeform"):
    """Return scored records indexed by (item, quant, condition)."""
    idx: dict[tuple[str, str, str], Scored] = {}
    items: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("variant", "standard") != variant or rec["protocol"] != protocol:
                continue
            idx[(rec["item_id"], rec["quant"], rec["condition"])] = score_record(rec)
            items.add(rec["item_id"])
    return idx, sorted(items)


def available_quants(idx) -> list[str]:
    return [q for q in ORDER if any(key[1] == q for key in idx)]


def capability(idx, items, quants) -> list[str]:
    """Common support: correct on the original at every compared level."""
    return [
        item for item in items
        if all(
            (item, q, "original") in idx
            and idx[(item, q, "original")].outcome == CORRECT
            for q in quants
        )
    ]


def load_coverage(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def level_curve(idx, items, quant: str, coverage: dict) -> list[CurvePoint]:
    points: list[CurvePoint] = []
    for level in LEVELS:
        condition = condition_for(level)
        observed = [
            item for item in items
            if (item, quant, condition) in idx
            and (level == 0 or item in coverage and str(level) in coverage[item]["levels"])
        ]
        if not observed:
            continue
        xs = [0.0 for _ in observed] if level == 0 else [
            float(coverage[item]["levels"][str(level)]) for item in observed
        ]
        y = _rate(idx[(item, quant, condition)].outcome in GROUNDED for item in observed)
        points.append(CurvePoint(level, statistics.median(xs), y, len(observed)))
    return points


def threshold50(points: list[CurvePoint]) -> float | None:
    """First interpolated crossing of grounded-response rate 0.5."""
    for a, b in zip(points, points[1:]):
        if a.grounded_rate < 0.5 <= b.grounded_rate:
            if b.grounded_rate == a.grounded_rate:
                return b.coverage
            weight = (0.5 - a.grounded_rate) / (b.grounded_rate - a.grounded_rate)
            return a.coverage + weight * (b.coverage - a.coverage)
    return None


def bootstrap_threshold(
    idx, items, quant: str, coverage: dict, n: int = 4000, seed: int = 0
) -> tuple[float, float]:
    if not items:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n):
        sample = [items[rng.randrange(len(items))] for _ in items]
        value = threshold50(level_curve(idx, sample, quant, coverage))
        if value is not None:
            draws.append(value)
    if len(draws) < n / 2:
        return float("nan"), float("nan")
    return _quantile(draws, 0.025), _quantile(draws, 0.975)


def logistic_threshold50(idx, items, quant: str, coverage: dict) -> float | None:
    """Sensitivity estimator: 50% point of an item-level logistic fit.

    Every (item, level) response is regressed on its own measured object
    coverage, so the estimate does not depend on linear interpolation between
    level medians.  Returns None when the fit is degenerate.
    """
    xs: list[float] = []
    ys: list[float] = []
    for item in items:
        for level in LEVELS:
            key = (item, quant, condition_for(level))
            if key not in idx:
                continue
            if level and not (item in coverage and str(level) in coverage[item]["levels"]):
                continue
            xs.append(0.0 if level == 0 else float(coverage[item]["levels"][str(level)]))
            ys.append(1.0 if idx[key].outcome in GROUNDED else 0.0)
    if not xs or len(set(ys)) < 2:
        return None
    b0 = b1 = 0.0
    for _ in range(100):  # Newton-Raphson on the logistic log-likelihood
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
            w = p * (1.0 - p)
            g0 += y - p
            g1 += (y - p) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if det <= 1e-12:
            return None
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (h00 * g1 - h01 * g0) / det
        b0 += d0
        b1 += d1
        if abs(d0) + abs(d1) < 1e-10:
            break
    return -b0 / b1 if b1 > 0 else None


def full_removal_metrics(idx, items, quant: str) -> dict[str, float]:
    target = [idx[(i, quant, "target_100")] for i in items]
    control = [idx[(i, quant, "control")] for i in items]
    ga = _rate(s.outcome in GROUNDED for s in target)
    fa = _rate(s.outcome in GROUNDED for s in control)
    return {
        "grounded_target": ga,
        "grounded_control": fa,
        "gai": ga - fa,
        "reference_leakage": _rate(s.reference_match for s in target),
        "control_accuracy": _rate(s.outcome == CORRECT for s in control),
    }


def bootstrap_gai(
    idx, items, quant: str, n: int = 4000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [items[rng.randrange(len(items))] for _ in items]
        draws.append(full_removal_metrics(idx, sample, quant)["gai"])
    return _quantile(draws, 0.025), _quantile(draws, 0.975)


def evaluate(path: str, coverage_path: str, variant="standard", protocol="freeform") -> dict:
    idx, items = load(path, variant, protocol)
    quants = available_quants(idx)
    support = capability(idx, items, quants)
    coverage = load_coverage(coverage_path)
    levels = {}
    for quant in quants:
        curve = level_curve(idx, support, quant, coverage)
        threshold = threshold50(curve)
        tlo, thi = bootstrap_threshold(idx, support, quant, coverage)
        glo, ghi = bootstrap_gai(idx, support, quant)
        levels[quant] = {
            "threshold50": threshold,
            "threshold50_ci95": [tlo, thi],
            "curve": [point.__dict__ for point in curve],
            "full_removal": {
                **full_removal_metrics(idx, support, quant),
                "gai_ci95": [glo, ghi],
            },
        }
    return {
        "path": os.path.basename(path),
        "variant": variant,
        "protocol": protocol,
        "items": len(items),
        "common_support": len(support),
        "quants": quants,
        "levels": levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="path", default=os.path.join(HERE, "results", "graded.jsonl"))
    parser.add_argument(
        "--coverage", default=os.path.join(HERE, "dataset", "probe_graded", "object_coverage.json")
    )
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--protocol", default="freeform")
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args()

    result = evaluate(args.path, args.coverage, args.variant, args.protocol)
    print(
        f"{result['items']} items | common support K={result['common_support']} | "
        f"{', '.join(result['quants'])}"
    )
    for quant, metrics in result["levels"].items():
        threshold = metrics["threshold50"]
        tlo, thi = metrics["threshold50_ci95"]
        full = metrics["full_removal"]
        curve = "  ".join(
            f"{p['coverage']:.3f}:{p['grounded_rate']:.3f}" for p in metrics["curve"]
        )
        print(f"{quant:>4}  {curve}")
        print(
            f"      thr50={threshold if threshold is not None else 'never'} "
            f"CI=[{tlo:.3f}, {thi:.3f}]  GAI@full={full['gai']:.3f} "
            f"CI=[{full['gai_ci95'][0]:.3f}, {full['gai_ci95'][1]:.3f}] "
            f"reference_leakage={full['reference_leakage']:.3f}"
        )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
