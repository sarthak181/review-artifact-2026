"""Generate release-paper figures from results/analysis_summary.json."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
with open(os.path.join(ROOT, "results", "analysis_summary.json"), encoding="utf-8") as fh:
    R = json.load(fh)

plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def benchmark_design():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))
    panels = [
        ("Original", "Answer object visible", False, False),
        ("Target intervention", "11--100% of answer object removed", True, False),
        ("Control intervention", "Different object removed", False, True),
    ]
    for ax, (title, subtitle, target_mask, control_mask) in zip(axes, panels):
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0.2, 0.3), 9.6, 5.7, facecolor="#f4f1ea", edgecolor="#555"))
        ax.add_patch(Rectangle((1.0, 1.0), 2.3, 1.5, facecolor="#91bfdb", edgecolor="#2166ac"))
        ax.add_patch(Rectangle((6.4, 3.3), 2.5, 1.7, facecolor="#fdae61", edgecolor="#b35806"))
        ax.text(2.15, 1.75, "answer\nobject", ha="center", va="center", fontsize=8)
        ax.text(7.65, 4.15, "other\nobject", ha="center", va="center", fontsize=8)
        if target_mask:
            ax.add_patch(Rectangle((1.0, 1.0), 2.3, 1.5, facecolor="#888", alpha=.88,
                                   hatch="////", edgecolor="#333"))
        if control_mask:
            ax.add_patch(Rectangle((6.4, 3.3), 2.5, 1.7, facecolor="#888", alpha=.88,
                                   hatch="////", edgecolor="#333"))
        ax.set_title(title, weight="bold", fontsize=9, pad=2)
        ax.text(5, 6.45, subtitle, ha="center", va="center", fontsize=7.5)
    fig.text(.5, .01,
             "Grounded response = explicit abstention or correct report that the queried object is absent.",
             ha="center", fontsize=8)
    save(fig, "benchmark_design")


def dose_response():
    systems = R["cross_model"]["systems"]
    styles = [
        ("Qwen2.5-VL-3B (bf16)", "#2166ac", "o"),
        ("Gemma-3-4B (int8)", "#b2182b", "s"),
        ("Qwen2.5-VL-7B (int4)", "#1b7837", "^")
    ]
    fig, ax = plt.subplots(figsize=(4.65, 3.25))
    for label, color, marker in styles:
        curve = systems[label]["curve"]
        ax.plot([p["coverage"] for p in curve], [p["grounded_rate"] for p in curve],
                marker=marker, ms=3.8, lw=1.5, color=color, label=label)
    ax.axhline(.5, color="#555", ls=":", lw=.9)
    ax.set(xlabel="Measured fraction of answer object removed",
           ylabel="Grounded-response rate", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(.99, .51, "50%", ha="right", va="bottom", color="#555", fontsize=7)
    ax.grid(axis="y", color="#ddd", lw=.5)
    save(fig, "dose_response_bf16")


def prompt_and_quantization():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"wspace": .34})
    ax = axes[0]
    variants = ["no_affordance", "standard", "last_resort"]
    labels = ["No abstention\naffordance", "Standard", "Abstain only as\nlast resort"]
    x = list(range(3)); width = .34
    for j, (model, color) in enumerate([
        ("qwen2.5-vl-3b_bf16", "#2166ac"), ("gemma-3-4b_int8", "#b2182b")
    ]):
        values = [R["prompt_variants"][model]["systems"][v]["full_removal"]["gai"]
                  for v in variants]
        ax.bar([v + (j-.5)*width for v in x], values, width, color=color,
               label="Qwen-3B" if j == 0 else "Gemma-4B")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("GAI at complete removal")
    ax.set_ylim(0, .7); ax.grid(axis="y", color="#ddd", lw=.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(a) Prompt is a major intervention", fontsize=9)

    ax = axes[1]
    primary = R["primary_quantization"]["levels"]
    quants = ["bf16", "int8", "int4"]
    values = [primary[q]["threshold50"] for q in quants]
    los = [v - primary[q]["threshold50_ci95"][0] for v, q in zip(values, quants)]
    his = [primary[q]["threshold50_ci95"][1] - v for v, q in zip(values, quants)]
    ax.errorbar(range(3), values, yerr=[los, his], fmt="o", color="#333", capsize=3, ms=5)
    ax.set_xticks(range(3), quants)
    ax.set_ylim(.65, .98); ax.set_ylabel("50% response threshold")
    ax.grid(axis="y", color="#ddd", lw=.5)
    ax.set_title("(b) No detected threshold degradation", fontsize=9)
    ax.text(1, .665, "item-bootstrap 95% CIs", ha="center", fontsize=7, color="#555")
    save(fig, "prompt_quantization")


if __name__ == "__main__":
    benchmark_design()
    dose_response()
    prompt_and_quantization()
