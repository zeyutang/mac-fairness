#!/usr/bin/env python3
"""Generate publication-quality figures from analysis CSV outputs.

Usage:
    python script/generate_figures.py                  # default: reads from figures/
    python script/generate_figures.py --results-dir figures --out-dir figures

Reads CSV files produced by analyze_results.py and generates:
  1. bbq_accuracy_by_model.pdf   — Grouped bar chart: accuracy per model × subcategory
  2. bbq_bias_heatmap.pdf        — Heatmap: disambig + ambig bias scores per model × subcategory
  3. bbq_accuracy_vs_bias.pdf    — Scatter: accuracy vs bias score per model
  4. agent_comparison.pdf         — Baseline vs 2-agent accuracy comparison (if 2-agent data exists)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_COLORS = {
    "gemma2-9b": "#4C72B0",
    "gemma2-27b": "#1f4e79",
    "qwen3-4b": "#DD8452",
    "qwen3-30b": "#c45e00",
    "ministral3-3b": "#55A868",
    "ministral3-8b": "#2d8043",
    "ministral3-14b": "#1b5c2e",
    "mistral03-7b": "#C44E52",
    "llama31-70b": "#8172B3",
    "llama32-3b": "#937860",
    "phi4": "#DA8BC3",
    "phi4-mini": "#8C8C8C",
    "glm47-flash": "#76B7B2",
}

SUBCAT_SHORT = {
    "bbq_age_sampled": "Age",
    "bbq_disability_status_sampled": "Disability",
    "bbq_gender_identity_sampled": "Gender",
    "bbq_nationality_sampled": "Nationality",
    "bbq_physical_appearance_sampled": "Appearance",
    "bbq_race_ethnicity_sampled": "Race/Eth.",
    "bbq_race_x_gender_sampled": "Race×Gender",
    "bbq_race_x_ses_sampled": "Race×SES",
    "bbq_religion_sampled": "Religion",
    "bbq_ses_sampled": "SES",
    "bbq_sexual_orientation_sampled": "Sex. Orient.",
}

MODEL_ORDER = [
    "ministral3-3b", "qwen3-4b", "mistral03-7b", "gemma2-9b",
    "ministral3-8b", "ministral3-14b", "gemma2-27b", "qwen3-30b", "llama31-70b",
]


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#888888")


def _short_subcat(subcat: str) -> str:
    return SUBCAT_SHORT.get(subcat, subcat.replace("bbq_", "").replace("_sampled", ""))


def _order_models(models: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    return known + unknown


# ── Figure 1: Accuracy grouped bar ───────────────────────────────────────────

def fig_accuracy_bars(df: pd.DataFrame, out_path: Path, title: str = "BBQ Accuracy by Model and Social Category"):
    """Grouped bar chart with error bars from rep std."""
    models = _order_models(list(df["model"].unique()))
    subcats = sorted(df["benchmark_subcategory"].unique(), key=lambda s: list(SUBCAT_SHORT.keys()).index(s) if s in SUBCAT_SHORT else 999)
    n_m, n_s = len(models), len(subcats)

    fig, ax = plt.subplots(figsize=(max(7, n_s * 0.9), 3.5))
    x = np.arange(n_s)
    total_w = 0.85
    bw = total_w / n_m
    offsets = np.linspace(-total_w / 2 + bw / 2, total_w / 2 - bw / 2, n_m)

    for mi, model in enumerate(models):
        means, stds = [], []
        for subcat in subcats:
            cell = df[(df["model"] == model) & (df["benchmark_subcategory"] == subcat)]
            vals = cell["overall_accuracy"].dropna()
            means.append(vals.mean() * 100 if len(vals) > 0 else float("nan"))
            stds.append(vals.std() * 100 if len(vals) > 1 else 0)

        ax.bar(x + offsets[mi], means, bw * 0.88, yerr=stds,
               label=model, color=_model_color(model), alpha=0.88,
               error_kw={"linewidth": 0.5, "capsize": 1.5, "capthick": 0.5})

    ax.axhline(100 / 3, color="gray", linestyle="--", linewidth=0.6, alpha=0.5, label="Random (33%)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_subcat(s) for s in subcats], rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3, framealpha=0.9, borderpad=0.3, handlelength=1.0)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── Figure 2: Bias score heatmap ─────────────────────────────────────────────

def fig_bias_heatmap(df: pd.DataFrame, out_path: Path):
    """Heatmap of disambig and ambig bias scores."""
    models = _order_models(list(df["model"].unique()))
    subcats = sorted(df["benchmark_subcategory"].unique(), key=lambda s: list(SUBCAT_SHORT.keys()).index(s) if s in SUBCAT_SHORT else 999)

    # Build matrices
    disambig_mat = np.full((len(models), len(subcats)), np.nan)
    ambig_mat = np.full((len(models), len(subcats)), np.nan)

    for mi, model in enumerate(models):
        for si, subcat in enumerate(subcats):
            cell = df[(df["model"] == model) & (df["benchmark_subcategory"] == subcat)]
            d = cell["disambig_bias_score"].dropna()
            a = cell["ambig_bias_score"].dropna()
            if len(d) > 0:
                disambig_mat[mi, si] = d.mean()
            if len(a) > 0:
                ambig_mat[mi, si] = a.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(2.5, len(models) * 0.45)),
                                    gridspec_kw={"wspace": 0.35})

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

    for ax, mat, title in [(ax1, disambig_mat, "Disambiguated Bias Score"),
                           (ax2, ambig_mat, "Ambiguous Bias Score")]:
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(subcats)))
        ax.set_xticklabels([_short_subcat(s) for s in subcats], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(title)

        # Annotate cells
        for mi in range(len(models)):
            for si in range(len(subcats)):
                val = mat[mi, si]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.25 else "black"
                    ax.text(si, mi, f"{val:+.2f}", ha="center", va="center",
                            fontsize=6, color=color)

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.7, label="Bias Score", pad=0.02)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── Figure 3: Accuracy vs Bias scatter ───────────────────────────────────────

def fig_accuracy_vs_bias(df: pd.DataFrame, out_path: Path):
    """Scatter plot of accuracy vs disambig bias score per (model, subcategory)."""
    fig, ax = plt.subplots(figsize=(5, 4))

    models = _order_models(list(df["model"].unique()))
    subcats = sorted(df["benchmark_subcategory"].unique())

    for model in models:
        accs, biases = [], []
        for subcat in subcats:
            cell = df[(df["model"] == model) & (df["benchmark_subcategory"] == subcat)]
            a = cell["overall_accuracy"].dropna()
            b = cell["disambig_bias_score"].dropna()
            if len(a) > 0 and len(b) > 0:
                accs.append(a.mean() * 100)
                biases.append(b.mean())

        ax.scatter(biases, accs, label=model, color=_model_color(model), s=25, alpha=0.8, edgecolors="white", linewidths=0.3)

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Disambiguated Bias Score")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Accuracy vs. Bias by Model")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── Figure 4: Baseline vs 2-Agent comparison ────────────────────────────────

def fig_agent_comparison(baseline_df: pd.DataFrame, id_df: pd.DataFrame, va_df: pd.DataFrame, out_path: Path):
    """Compare baseline vs identity-agent vs vanilla-agent accuracy.

    One subplot per overlapping model, with 3 bars (baseline, identity, vanilla)
    per subcategory.
    """
    common_models = _order_models(sorted(set(baseline_df["model"]) & set(id_df["model"])))
    if not common_models:
        print("  No overlapping models between baseline and 2-agent, skipping agent_comparison")
        return

    common_subcats = sorted(set(baseline_df["benchmark_subcategory"]) & set(id_df["benchmark_subcategory"]),
                            key=lambda s: list(SUBCAT_SHORT.keys()).index(s) if s in SUBCAT_SHORT else 999)
    n_s = len(common_subcats)
    n_m = len(common_models)

    fig, axes = plt.subplots(1, n_m, figsize=(max(4, n_s * 0.55) * n_m, 3.5), sharey=True, squeeze=False)
    axes = axes[0]

    cond_colors = {"Baseline": "#4C72B0", "Identity Agent": "#DD8452", "Vanilla Agent": "#55A868"}
    x = np.arange(n_s)
    bw = 0.25

    for mi, model in enumerate(common_models):
        ax = axes[mi]
        for ci, (cond, src) in enumerate([("Baseline", baseline_df), ("Identity Agent", id_df), ("Vanilla Agent", va_df)]):
            vals = []
            for subcat in common_subcats:
                cell = src[(src["model"] == model) & (src["benchmark_subcategory"] == subcat)]
                v = cell["overall_accuracy"].dropna()
                vals.append(v.mean() * 100 if len(v) > 0 else float("nan"))

            offset = (ci - 1) * bw
            ax.bar(x + offset, vals, bw * 0.9, label=cond if mi == 0 else None,
                   color=cond_colors[cond], alpha=0.85, edgecolor="white", linewidth=0.3)

        ax.axhline(100 / 3, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_subcat(s) for s in common_subcats], rotation=45, ha="right")
        ax.set_title(model, fontsize=9)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if mi == 0:
            ax.set_ylabel("Accuracy (%)")

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Baseline vs. 2-Agent Accuracy", fontsize=11, y=1.06)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures from analysis CSVs")
    parser.add_argument("--results-dir", default="figures", help="Root directory with analysis CSVs")
    parser.add_argument("--out-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Output format")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format

    # Load baseline metrics
    baseline_path = results_dir / "baseline" / "bbq_metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        print(f"Loaded baseline: {len(baseline)} rows, models={sorted(baseline['model'].unique())}")

        fig_accuracy_bars(baseline, out_dir / f"bbq_accuracy_by_model.{ext}",
                          title="BBQ Accuracy by Model and Social Category (1-Agent Baseline)")
        fig_bias_heatmap(baseline, out_dir / f"bbq_bias_heatmap.{ext}")
        fig_accuracy_vs_bias(baseline, out_dir / f"bbq_accuracy_vs_bias.{ext}")
    else:
        print(f"  No baseline metrics found at {baseline_path}")
        baseline = pd.DataFrame()

    # Load 2-agent metrics
    id_path = results_dir / "2agent_identity" / "bbq_metrics.csv"
    va_path = results_dir / "2agent_vanilla" / "bbq_metrics.csv"
    if id_path.exists() and va_path.exists():
        id_df = pd.read_csv(id_path)
        va_df = pd.read_csv(va_path)
        print(f"Loaded 2-agent identity: {len(id_df)} rows, vanilla: {len(va_df)} rows")

        fig_accuracy_bars(id_df, out_dir / f"bbq_accuracy_2agent_identity.{ext}",
                          title="BBQ Accuracy — 2-Agent Identity Agent")
        fig_accuracy_bars(va_df, out_dir / f"bbq_accuracy_2agent_vanilla.{ext}",
                          title="BBQ Accuracy — 2-Agent Vanilla Agent")

        if not baseline.empty:
            fig_agent_comparison(baseline, id_df, va_df, out_dir / f"bbq_agent_comparison.{ext}")
    else:
        print(f"  No 2-agent metrics found")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
