"""Recompute all release-paper statistics from the shipped JSONL results.

The script writes a machine-readable audit trail to
``results/analysis_summary.json``.  Every comparison uses an explicitly stated
common capability support: items answered correctly in the original image by
every system or prompt being compared.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict

import graded_eval as G

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "dataset", "probe_graded")
COVERAGE_PATH = os.path.join(DATA, "object_coverage.json")


def path(name: str) -> str:
    return os.path.join(RESULTS, name)


def json_clean(value):
    """Represent undefined estimates as JSON null, never non-standard NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    return value


def common_items(systems):
    """Common original-correct support across (index, quant) systems."""
    item_sets = []
    for idx, quant in systems:
        item_sets.append({
            item for item, q, condition in idx
            if q == quant and condition == "original"
            and idx[(item, q, condition)].outcome == G.CORRECT
        })
    return sorted(set.intersection(*item_sets)) if item_sets else []


def system_summary(idx, items, quant, coverage, boot_seed=0):
    curve = G.level_curve(idx, items, quant, coverage)
    full = G.full_removal_metrics(idx, items, quant)
    full["gai_ci95"] = list(G.bootstrap_gai(idx, items, quant, seed=boot_seed))
    return {
        "n": len(items),
        "original_accuracy_all": None,
        "threshold50": G.threshold50(curve),
        "threshold50_ci95": list(
            G.bootstrap_threshold(idx, items, quant, coverage, seed=boot_seed)
        ),
        "curve": [p.__dict__ for p in curve],
        "full_removal": full,
    }


def original_accuracy(idx, quant):
    rows = [s for (item, q, condition), s in idx.items()
            if q == quant and condition == "original"]
    return sum(s.outcome == G.CORRECT for s in rows) / len(rows)


def paired_threshold_delta(idx, items, qa, qb, coverage, n=4000, seed=71):
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [items[rng.randrange(len(items))] for _ in items]
        a = G.threshold50(G.level_curve(idx, sample, qa, coverage))
        b = G.threshold50(G.level_curve(idx, sample, qb, coverage))
        if a is not None and b is not None:
            draws.append(b - a)
    return {
        "estimate": G.threshold50(G.level_curve(idx, items, qb, coverage))
                    - G.threshold50(G.level_curve(idx, items, qa, coverage)),
        "ci95": [G._quantile(draws, 0.025), G._quantile(draws, 0.975)],
        "defined_bootstrap_draws": len(draws),
    }


def prompt_analysis(coverage):
    out = {}
    configurations = {
        "qwen2.5-vl-3b_bf16": [
            ("no_affordance", path("graded_prompts.jsonl"), "bf16"),
            ("standard", path("graded.jsonl"), "bf16"),
            ("last_resort", path("graded_prompts.jsonl"), "bf16"),
        ],
        "gemma-3-4b_int8": [
            ("no_affordance", path("graded_gemma.jsonl"), "int8"),
            ("standard", path("graded_gemma.jsonl"), "int8"),
            ("last_resort", path("graded_gemma.jsonl"), "int8"),
        ],
    }
    for model, configs in configurations.items():
        loaded = []
        for variant, filename, quant in configs:
            idx, _ = G.load(filename, variant=variant)
            loaded.append((variant, idx, quant))
        support = common_items([(idx, quant) for _, idx, quant in loaded])
        systems = {}
        for seed, (variant, idx, quant) in enumerate(loaded, start=101):
            summary = system_summary(idx, support, quant, coverage, seed)
            summary["original_accuracy_all"] = original_accuracy(idx, quant)
            systems[variant] = summary
        out[model] = {"common_support": len(support), "systems": systems}
    return out


def cross_model_analysis(coverage):
    configs = [
        ("Qwen2.5-VL-3B (bf16)", path("graded.jsonl"), "bf16"),
        ("Gemma-3-4B (int8)", path("graded_gemma.jsonl"), "int8"),
        ("Qwen2.5-VL-7B (int4)", path("graded_7b.jsonl"), "int4"),
    ]
    loaded = []
    for label, filename, quant in configs:
        idx, _ = G.load(filename, variant="standard")
        loaded.append((label, idx, quant))
    support = common_items([(idx, quant) for _, idx, quant in loaded])
    systems = {}
    for seed, (label, idx, quant) in enumerate(loaded, start=201):
        summary = system_summary(idx, support, quant, coverage, seed)
        summary["original_accuracy_all"] = original_accuracy(idx, quant)
        systems[label] = summary
    return {
        "common_support": len(support),
        "systems": systems,
        "note": (
            "Systems use their stated deployment precisions. Precision effects are "
            "analyzed separately within Qwen-3B."
        ),
    }


def control_sensitivity(idx, support, quant):
    manifest = {m["item_id"]: m for m in json.load(
        open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))}

    def ratio(m):
        a, b = float(m["masked_frac_100"]), float(m["control_masked_frac"])
        return max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")

    def named(m):
        question = re.sub(r"[^a-z0-9 ]", " ", m["question"].lower())
        category = re.sub(r"[^a-z0-9 ]", " ", m["control_category"].lower()).strip()
        return bool(category) and re.search(r"\b" + re.escape(category) + r"\b", question) is not None

    valid = [i for i in support if i in manifest]
    subsets = {
        "all_common_support": valid,
        "area_ratio_le_1.5": [i for i in valid if ratio(manifest[i]) <= 1.5],
        "control_not_named": [i for i in valid if not named(manifest[i])],
        "both_filters": [i for i in valid if ratio(manifest[i]) <= 1.5 and not named(manifest[i])],
    }
    ratios = sorted(ratio(manifest[i]) for i in valid)
    return {
        "painted_area_ratio": {
            "median": G._quantile(ratios, 0.5),
            "p90": G._quantile(ratios, 0.9),
            "fraction_le_1.5": sum(x <= 1.5 for x in ratios) / len(ratios),
            "fraction_le_2": sum(x <= 2 for x in ratios) / len(ratios),
        },
        "control_category_named_fraction": sum(named(manifest[i]) for i in valid) / len(valid),
        "subsets": {
            label: {"n": len(items), **G.full_removal_metrics(idx, items, quant)}
            for label, items in subsets.items() if items
        },
        "caveat": (
            "The binary control preserves the answer object and is matched only approximately "
            "in painted area. The graded partial-removal conditions have no dose-matched controls."
        ),
    }


def paired_bootstrap(stat, items, n=4000, seed=0):
    """Point estimate and percentile 95% CI of stat(items) under item resampling."""
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        value = stat([items[rng.randrange(len(items))] for _ in items])
        if value is not None:
            draws.append(value)
    return {
        "estimate": stat(items),
        "ci95": [G._quantile(draws, 0.025), G._quantile(draws, 0.975)],
        "defined_bootstrap_draws": len(draws),
    }


def grounded_rate(idx, items, quant, condition):
    return sum(idx[(i, quant, condition)].outcome in G.GROUNDED for i in items) / len(items)


def logistic_delta(idx, qa, qb, coverage):
    def stat(items):
        a = G.logistic_threshold50(idx, items, qa, coverage)
        b = G.logistic_threshold50(idx, items, qb, coverage)
        return None if a is None or b is None else b - a
    return stat


def precision_matched_models(coverage):
    """Model comparisons at a shared precision (Table 1 mixes model and precision)."""
    q3, _ = G.load(path("graded.jsonl"))
    pairs = {
        "int8": [("Qwen2.5-VL-3B (int8)", q3), ("Gemma-3-4B (int8)", G.load(path("graded_gemma.jsonl"))[0])],
        "int4": [("Qwen2.5-VL-3B (int4)", q3), ("Qwen2.5-VL-7B (int4)", G.load(path("graded_7b.jsonl"))[0])],
    }
    out = {}
    for quant, systems in pairs.items():
        support = common_items([(idx, quant) for _, idx in systems])
        summaries = {}
        for seed, (label, idx) in enumerate(systems, start=301):
            summary = system_summary(idx, support, quant, coverage, seed)
            summary["original_accuracy_all"] = original_accuracy(idx, quant)
            summaries[label] = summary
        (_, ia), (label_b, ib) = systems
        gai = lambda s: (G.full_removal_metrics(ib, s, quant)["gai"]
                         - G.full_removal_metrics(ia, s, quant)["gai"])
        out[quant] = {
            "common_support": len(support),
            "systems": summaries,
            f"gai_{label_b}_minus_qwen3b": paired_bootstrap(gai, support, seed=311),
        }
    return out


def area_matched_control(systems, support, coverage):
    """Target dose painting no more image area than the control, versus control.

    For each item the matched dose is the largest target dose whose painted image
    fraction is <= the control's; selection uses mask geometry only.  If grounded
    responses still exceed the control rate, painted area alone cannot explain them.
    """
    manifest = {m["item_id"]: m for m in json.load(
        open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))}

    def matched(item):
        m = manifest[item]
        ok = [d for d in G.LEVELS[1:]
              if float(m[f"masked_frac_{d}"]) <= float(m["control_masked_frac"])]
        return ok[-1] if ok else None

    doses = {i: matched(i) for i in support if i in manifest}
    subsets = {
        "all": [i for i, d in doses.items() if d is not None],
        "partial_doses_only": [i for i, d in doses.items() if d is not None and d < 100],
    }
    ratios = sorted(float(manifest[i][f"masked_frac_{doses[i]}"])
                    / float(manifest[i]["control_masked_frac"]) for i in subsets["all"])
    out = {
        "support": len(support),
        "matched_dose_counts": {str(d): sum(v == d for v in doses.values()) for d in G.LEVELS[1:]},
        "median_target_over_control_area": G._quantile(ratios, 0.5),
        "median_matched_object_coverage": G._quantile(sorted(
            float(coverage[i]["levels"][str(doses[i])]) for i in subsets["all"]), 0.5),
        "systems": {},
    }
    for seed, (label, idx, quant) in enumerate(systems, start=401):
        gr = lambda s: sum(idx[(i, quant, f"target_{doses[i]}")].outcome in G.GROUNDED
                           for i in s) / len(s)
        out["systems"][label] = {
            name: {
                "n": len(items),
                "grounded_target_matched": gr(items),
                "grounded_control": grounded_rate(idx, items, quant, "control"),
                "difference": paired_bootstrap(
                    lambda s: gr(s) - grounded_rate(idx, s, quant, "control"), items, seed=seed),
            }
            for name, items in subsets.items()
        }
    return out


def estimator_sensitivity(coverage, systems, support, q3, q3_support):
    """Logistic thresholds and fixed-dose rates, which avoid linear interpolation."""
    thresholds = {}
    for seed, (label, idx, quant) in enumerate(systems, start=501):
        thresholds[label] = paired_bootstrap(
            lambda s: G.logistic_threshold50(idx, s, quant, coverage), support, seed=seed)
    quant_thresholds = {
        q: paired_bootstrap(lambda s: G.logistic_threshold50(q3, s, q, coverage),
                            q3_support, seed=511 + k)
        for k, q in enumerate(G.ORDER)
    }
    deltas = {
        f"{q}_minus_bf16": paired_bootstrap(logistic_delta(q3, "bf16", q, coverage),
                                            q3_support, seed=521 + k)
        for k, q in enumerate(("int8", "int4"))
    }
    fixed = {
        f"target_{level}": {
            f"{q}_minus_bf16": paired_bootstrap(
                lambda s: (grounded_rate(q3, s, q, f"target_{level}")
                           - grounded_rate(q3, s, "bf16", f"target_{level}")),
                q3_support, seed=531 + level)
            for q in ("int8", "int4")
        }
        for level in (54, 73, 100)
    }
    return {
        "note": ("Item-level logistic regression of grounded response on measured object "
                 "coverage; the 50% point may extrapolate beyond 1.0 when a curve never "
                 "reaches 50%."),
        "cross_model_logistic_threshold50": thresholds,
        "qwen3b_logistic_threshold50": quant_thresholds,
        "qwen3b_logistic_paired_deltas": deltas,
        "qwen3b_fixed_dose_grounded_rate_deltas": fixed,
    }


def main():
    coverage = G.load_coverage(COVERAGE_PATH)
    primary = G.evaluate(path("graded.jsonl"), COVERAGE_PATH)
    heldout = G.evaluate(
        path("graded_heldout.jsonl"),
        os.path.join(ROOT, "dataset", "probe_graded_heldout", "object_coverage.json"),
    )
    idx, items = G.load(path("graded.jsonl"))
    support = G.capability(idx, items, G.available_quants(idx))
    summary = {
        "schema_version": 1,
        "estimator": (
            "One grounded-response rate per nominal intervention level, positioned at median "
            "measured object coverage; linear first crossing; item bootstrap."
        ),
        "primary_quantization": primary,
        "heldout_quantization": heldout,
        "paired_threshold_deltas": {
            "int8_minus_bf16": paired_threshold_delta(idx, support, "bf16", "int8", coverage),
            "int4_minus_bf16": paired_threshold_delta(idx, support, "bf16", "int4", coverage),
        },
        "cross_model": cross_model_analysis(coverage),
        "prompt_variants": prompt_analysis(coverage),
        "control_sensitivity_bf16": control_sensitivity(idx, support, "bf16"),
    }
    cross = [
        ("Qwen2.5-VL-3B (bf16)", idx, "bf16"),
        ("Gemma-3-4B (int8)", G.load(path("graded_gemma.jsonl"))[0], "int8"),
        ("Qwen2.5-VL-7B (int4)", G.load(path("graded_7b.jsonl"))[0], "int4"),
    ]
    cross_support = common_items([(i, q) for _, i, q in cross])
    summary["robustness"] = {
        "precision_matched_models": precision_matched_models(coverage),
        "area_matched_control_cross_model": area_matched_control(cross, cross_support, coverage),
        "area_matched_control_qwen3b": area_matched_control(
            [(f"Qwen2.5-VL-3B ({q})", idx, q) for q in G.ORDER], support, coverage),
        "threshold_estimator_sensitivity": estimator_sensitivity(
            coverage, cross, cross_support, idx, support),
    }
    summary = json_clean(summary)
    output = path("analysis_summary.json")
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print("wrote", output)


if __name__ == "__main__":
    main()
