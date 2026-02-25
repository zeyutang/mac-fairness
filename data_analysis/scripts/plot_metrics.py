"""Generate visualizations from computed metric tables.

One-agent mode outputs:
- model-level bar charts (accuracy, instability),
- model tradeoff scatter plots.

Multi-agent mode outputs:
- model-level behavior bar charts from aggregated metrics.

If matplotlib is available, plots are rendered with richer styling.
Otherwise a built-in SVG fallback renderer is used.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

from _common import cfg_path, load_config, parse_args

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def _palette(n: int) -> list[str]:
    base = [
        "#2f6f89",
        "#bc6c25",
        "#3a7d44",
        "#7a5195",
        "#c1121f",
        "#577590",
        "#ff7f51",
        "#6d597a",
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], colors: list[str] | None = None) -> None:
    width = 1020
    height = 520
    left = 240
    top = 88
    bar_h = 28
    gap = 16
    max_w = width - left - 110

    max_val = max(values) if values else 1.0
    if max_val <= 0:
        max_val = 1.0

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<defs>",
        "<linearGradient id='bg' x1='0%' y1='0%' x2='100%' y2='100%'>",
        "<stop offset='0%' stop-color='#f7f9fc'/>",
        "<stop offset='100%' stop-color='#eef2f7'/>",
        "</linearGradient>",
        "</defs>",
        "<rect width='100%' height='100%' fill='url(#bg)' />",
        "<rect x='16' y='14' width='988' height='492' rx='16' fill='white' stroke='#d9e1ec'/>",
        f"<text x='36' y='48' font-size='22' font-family='Helvetica' fill='#1d2939'>{title}</text>",
        "<text x='36' y='70' font-size='12' font-family='Helvetica' fill='#667085'>Each bar is one model</text>",
    ]

    palette = colors if colors else _palette(len(labels))
    for i in range(5):
        gx = left + int((max_w / 4.0) * i)
        lines.append(f"<line x1='{gx}' y1='{top - 10}' x2='{gx}' y2='{height - 46}' stroke='#edf2f7'/>")

    y = top
    for idx, (label, value) in enumerate(zip(labels, values)):
        bar_w = int((value / max_val) * max_w)
        color = palette[idx % len(palette)]
        lines.append(
            f"<text x='36' y='{y + 20}' font-size='14' font-family='Helvetica' fill='#344054'>{label}</text>"
        )
        lines.append(
            f"<rect x='{left}' y='{y}' width='{bar_w}' height='{bar_h}' fill='{color}' rx='8' ry='8'/>"
        )
        lines.append(
            f"<text x='{left + bar_w + 10}' y='{y + 20}' font-size='13' font-family='Helvetica' fill='#101828'>{value:.3f}</text>"
        )
        y += bar_h + gap

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _scale(val: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2.0
    frac = (val - lo) / (hi - lo)
    return out_lo + frac * (out_hi - out_lo)


def write_scatter_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[str, float, float]],
) -> None:
    width = 1060
    height = 620
    left = 110
    right = 220
    top = 84
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [p[1] for p in points] or [0.0]
    ys = [p[2] for p in points] or [0.0]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_pad = max((x_max - x_min) * 0.1, 0.02)
    y_pad = max((y_max - y_min) * 0.1, 0.02)
    x0 = x_min - x_pad
    x1 = x_max + x_pad
    y0 = y_min - y_pad
    y1 = y_max + y_pad

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<defs>",
        "<linearGradient id='bg' x1='0%' y1='0%' x2='100%' y2='100%'>",
        "<stop offset='0%' stop-color='#f7f9fc'/>",
        "<stop offset='100%' stop-color='#eef2f7'/>",
        "</linearGradient>",
        "</defs>",
        "<rect width='100%' height='100%' fill='url(#bg)' />",
        "<rect x='16' y='14' width='1028' height='592' rx='16' fill='white' stroke='#d9e1ec'/>",
        f"<text x='36' y='48' font-size='22' font-family='Helvetica' fill='#1d2939'>{html.escape(title)}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#475467' stroke-width='1.4'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#475467' stroke-width='1.4'/>",
    ]

    for i in range(5):
        xv = x0 + (x1 - x0) * i / 4.0
        x = _scale(xv, x0, x1, left, left + plot_w)
        lines.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}' stroke='#eef2f6'/>")
        lines.append(f"<line x1='{x:.1f}' y1='{top + plot_h}' x2='{x:.1f}' y2='{top + plot_h + 6}' stroke='#667085'/>")
        lines.append(f"<text x='{x - 18:.1f}' y='{top + plot_h + 24}' font-size='12' font-family='Helvetica' fill='#475467'>{xv:.2f}</text>")
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4.0
        y = _scale(yv, y0, y1, top + plot_h, top)
        lines.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#eef2f6'/>")
        lines.append(f"<line x1='{left - 6}' y1='{y:.1f}' x2='{left}' y2='{y:.1f}' stroke='#667085'/>")
        lines.append(f"<text x='{left - 62}' y='{y + 4:.1f}' font-size='12' font-family='Helvetica' fill='#475467'>{yv:.2f}</text>")

    palette = _palette(len(points))
    for idx, (label, xv, yv) in enumerate(points):
        cx = _scale(xv, x0, x1, left, left + plot_w)
        cy = _scale(yv, y0, y1, top + plot_h, top)
        color = palette[idx % len(palette)]
        lines.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6.8' fill='{color}' fill-opacity='0.92' stroke='white' stroke-width='1.2'/>")
        lines.append(
            f"<text x='{cx + 9:.1f}' y='{cy - 8:.1f}' font-size='12' font-family='Helvetica' fill='#1d2939'>{html.escape(label)}</text>"
        )

    legend_x = left + plot_w + 24
    legend_y = top + 8
    lines.append(f"<text x='{legend_x}' y='{legend_y - 6}' font-size='12' font-family='Helvetica' fill='#475467'>Model Colors</text>")
    for idx, (label, _, _) in enumerate(points):
        ly = legend_y + idx * 20 + 14
        color = palette[idx % len(palette)]
        lines.append(f"<rect x='{legend_x}' y='{ly - 10}' width='12' height='12' rx='2' fill='{color}'/>")
        lines.append(f"<text x='{legend_x + 18}' y='{ly}' font-size='11' font-family='Helvetica' fill='#344054'>{html.escape(label)}</text>")

    lines.extend(
        [
            f"<text x='{left + plot_w / 2 - 110:.1f}' y='{height - 34}' font-size='14' font-family='Helvetica' fill='#344054'>{html.escape(x_label)}</text>",
            f"<text transform='translate(28,{top + plot_h / 2:.1f}) rotate(-90)' font-size='14' font-family='Helvetica' fill='#344054'>{html.escape(y_label)}</text>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_one_agent_with_matplotlib(
    plots_dir: Path,
    summary_rows: list[dict[str, str]],
) -> int:
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "#eef2f7",
            "axes.edgecolor": "#94a3b8",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.color": "#94a3b8",
        }
    )
    models = [r["model"] for r in summary_rows]
    accuracy = [float(r["accuracy_rate"]) for r in summary_rows]
    instability = [float(r["instability_rate"]) for r in summary_rows]
    mean_disambig = [float(r["mean_disambig_bias_score"]) for r in summary_rows]
    mean_ambig = [float(r["mean_ambig_bias_score"]) for r in summary_rows]

    # Accuracy bar.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = _palette(len(models))
    ax.bar(models, accuracy, color=colors)
    ax.set_title("One-Agent Accuracy Rate by Model")
    ax.set_ylabel("Accuracy Rate")
    ax.set_ylim(0.0, 1.0)
    for i, v in enumerate(accuracy):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#1f2937")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "one_agent_accuracy_rate.svg")
    plt.close(fig)

    # Instability bar.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(models, instability, color=colors)
    ax.set_title("One-Agent Instability Rate by Model")
    ax.set_ylabel("Instability Rate")
    ax.set_ylim(0.0, min(1.0, max(instability + [0.4]) + 0.05))
    for i, v in enumerate(instability):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#1f2937")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "one_agent_instability_rate.svg")
    plt.close(fig)

    # Accuracy vs instability scatter.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.scatter(accuracy, instability, c=colors, s=95, alpha=0.95, edgecolors="white", linewidths=1.0)
    for name, x, y, c in zip(models, accuracy, instability, colors):
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    for model, color in zip(models, colors):
        ax.scatter([], [], color=color, label=model, s=55)
    ax.set_title("Model Tradeoff: Accuracy vs Instability")
    ax.set_xlabel("Accuracy Rate")
    ax.set_ylabel("Instability Rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Model")
    fig.tight_layout()
    fig.savefig(plots_dir / "one_agent_accuracy_vs_instability_scatter.svg")
    plt.close(fig)

    # Disambig bias vs ambig bias scatter.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.scatter(mean_disambig, mean_ambig, c=colors, s=95, alpha=0.95, edgecolors="white", linewidths=1.0)
    for name, x, y in zip(models, mean_disambig, mean_ambig):
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    for model, color in zip(models, colors):
        ax.scatter([], [], color=color, label=model, s=55)
    # ax.axhline(0.0, color="#888", linewidth=1.0, linestyle="--")
    # ax.axvline(0.0, color="#888", linewidth=1.0, linestyle="--")
    ax.set_title("Model Bias Profile: Disambig vs Ambig Bias")
    ax.set_xlabel("Mean Disambig Bias Score")
    ax.set_ylabel("Mean Ambig Bias Score")
    # ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Model")
    fig.tight_layout()
    fig.savefig(plots_dir / "one_agent_bias_tradeoff_scatter.svg")
    plt.close(fig)

    return 4


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()

    if schema_profile == "one_agent":
        summary_path = cfg_path(config, "one_agent_summary_output", "metrics/{run_label}/one_agent_summary_by_model.csv")
        plots_dir = cfg_path(config, "plots_dir", "plots/{run_label}")
        if not summary_path.exists():
            print("plot_metrics: skipped one-agent plots because one-agent summary file is missing")
            return 0
        plots_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print("plot_metrics: skipped one-agent plots because summary has no rows")
            return 0
        labels = [r["model"] for r in rows]
        accuracy = [float(r["accuracy_rate"]) for r in rows]
        instability = [float(r["instability_rate"]) for r in rows]
        if MATPLOTLIB_AVAILABLE:
            files = _plot_one_agent_with_matplotlib(plots_dir, rows)
            print(f"plot_metrics: output_dir={plots_dir} files={files} (one-agent, matplotlib)")
            return 0

        print("plot_metrics: matplotlib not available; using basic SVG fallback. Install with: uv add matplotlib")
        write_bar_svg(
            plots_dir / "one_agent_accuracy_rate.svg",
            "One-Agent Accuracy Rate by Model",
            labels,
            accuracy,
            colors=_palette(len(labels)),
        )
        write_bar_svg(
            plots_dir / "one_agent_instability_rate.svg",
            "One-Agent Instability Rate by Model",
            labels,
            instability,
            colors=_palette(len(labels)),
        )
        scatter_1_points = [(r["model"], float(r["accuracy_rate"]), float(r["instability_rate"])) for r in rows]
        write_scatter_svg(
            plots_dir / "one_agent_accuracy_vs_instability_scatter.svg",
            "Model Tradeoff: Accuracy vs Instability",
            "Accuracy Rate",
            "Instability Rate",
            scatter_1_points,
        )
        scatter_2_points = [(r["model"], float(r["mean_disambig_bias_score"]), float(r["mean_ambig_bias_score"])) for r in rows]
        write_scatter_svg(
            plots_dir / "one_agent_bias_tradeoff_scatter.svg",
            "Model Bias Profile: Disambig vs Ambig Bias",
            "Mean Disambig Bias Score",
            "Mean Ambig Bias Score",
            scatter_2_points,
        )
        print(f"plot_metrics: output_dir={plots_dir} files=4 (one-agent, fallback)")
        return 0

    aggregate_path = cfg_path(config, "aggregate_output", "metrics/{run_label}/behavior_metrics_by_model.csv")
    plots_dir = cfg_path(config, "plots_dir", "plots/{run_label}")
    if schema_profile == "one_agent":
        print("plot_metrics: skipped for schema_profile=one_agent")
        return 0
    if not aggregate_path.exists():
        print("plot_metrics: skipped because aggregate metrics file is missing")
        return 0
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        print("plot_metrics: skipped because aggregate metrics has no rows")
        return 0

    labels = [r["model"] for r in rows]

    corrected = [float(r["avg_corrected_after_wrong_rate"]) for r in rows]
    write_bar_svg(
        plots_dir / "avg_corrected_after_wrong_rate.svg",
        "Average Corrected-After-Wrong Rate by Model",
        labels,
        corrected,
    )

    stood_firm = [float(r["avg_stood_firm_wrong_rate"]) for r in rows]
    write_bar_svg(
        plots_dir / "avg_stood_firm_wrong_rate.svg",
        "Average Stood-Firm-Wrong Rate by Model",
        labels,
        stood_firm,
    )

    print(f"plot_metrics: output_dir={plots_dir} files=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
