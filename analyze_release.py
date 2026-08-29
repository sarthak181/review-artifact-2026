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
    summary = json_clean(summary)
    output = path("analysis_summary.json")
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print("wrote", output)


if __name__ == "__main__":
    main()
