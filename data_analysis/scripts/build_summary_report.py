"""Build a short markdown summary of a pipeline run.

This script reads whichever artifacts exist for the active schema profile and
writes one report file with:
- run metadata,
- key headline metric,
- artifact paths produced in this run.

Output:
- `reports/{run_label}/summary.md`
"""

from __future__ import annotations

import csv

from _common import cfg_path, ensure_parent, load_config, parse_args


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    metrics_path = cfg_path(config, "metrics_output", "metrics/{run_label}/behavior_metrics.csv")
    aggregate_path = cfg_path(config, "aggregate_output", "metrics/{run_label}/behavior_metrics_by_model.csv")
    normalized_path = cfg_path(config, "normalized_output", "intermediate/{run_label}/normalized.jsonl")
    perspective_question_path = cfg_path(
        config, "perspective_question_output", "metrics/{run_label}/perspective_question.csv"
    )
    perspective_summary_path = cfg_path(
        config, "perspective_summary_output", "metrics/{run_label}/perspective_summary.csv"
    )
    one_agent_summary_path = cfg_path(
        config, "one_agent_summary_output", "metrics/{run_label}/one_agent_summary_by_model.csv"
    )
    one_agent_legacy_dir = cfg_path(config, "one_agent_legacy_analysis_dir", "metrics/{run_label}/legacy_analysis")
    report_path = ensure_parent(str(cfg_path(config, "report_output", "reports/{run_label}/summary.md")))
    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()
    metrics_version = str(config.get("metrics_version", "v1"))

    total_groups = 0
    total_rows = 0
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_groups += 1
                total_rows += int(row["total_rows"])
    elif normalized_path.exists():
        with normalized_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    total_rows += 1

    top_model = "n/a"
    top_value = -1.0
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = float(row["avg_corrected_after_wrong_rate"])
                if value > top_value:
                    top_value = value
                    top_model = row["model"]

    behavior_summary = ""
    one_agent_extra: list[str] = []
    if schema_profile == "one_agent":
        best_acc_model = "n/a"
        best_acc = -1.0
        if one_agent_summary_path.exists():
            with one_agent_summary_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    acc = float(row["accuracy_rate"])
                    if acc > best_acc:
                        best_acc = acc
                        best_acc_model = row["model"]
        behavior_summary = (
            f"- Best model by one-agent accuracy rate: {best_acc_model} ({best_acc:.3f})"
            if best_acc_model != "n/a"
            else "- One-agent summary has no rows"
        )
        one_agent_extra.append(
            f"- One-agent summary metrics: `{one_agent_summary_path}`" if one_agent_summary_path.exists() else "- One-agent summary metrics: skipped"
        )
        one_agent_extra.append(
            (
                f"- Legacy-compatible condition analysis: `{one_agent_legacy_dir}`"
                if one_agent_legacy_dir.exists() and any(one_agent_legacy_dir.glob("*.csv"))
                else "- Legacy-compatible condition analysis: skipped"
            )
        )
    else:
        behavior_summary = (
            f"- Best model by avg corrected-after-wrong rate: {top_model} ({top_value:.3f})"
            if top_model != "n/a"
            else "- No model rows found"
        )

    plot_dir = cfg_path(config, "plots_dir", "plots/{run_label}")
    plot_status = (
        f"- Plots folder: `{plot_dir}`"
        if plot_dir.exists()
        else "- Plots skipped (no aggregate behavior metrics)"
    )

    report = [
        "# Analysis Summary",
        "",
        f"- Run label: {config.get('run_label', 'default')}",
        f"- Metrics version: {metrics_version}",
        f"- Schema profile: {config.get('schema_profile', 'auto')}",
        f"- Group rows analyzed: {total_groups}",
        f"- Normalized records analyzed: {total_rows}",
        behavior_summary,
        "",
        "## Artifacts",
        "",
        (
            f"- Detailed behavior metrics: `{metrics_path}`"
            if metrics_path.exists()
            else "- Detailed behavior metrics: skipped"
        ),
        (
            f"- Aggregated behavior metrics: `{aggregate_path}`"
            if aggregate_path.exists()
            else "- Aggregated behavior metrics: skipped"
        ),
        (
            f"- Perspective per-question metrics: `{perspective_question_path}`"
            if perspective_question_path.exists()
            else "- Perspective per-question metrics: skipped"
        ),
        (
            f"- Perspective summary metrics: `{perspective_summary_path}`"
            if perspective_summary_path.exists()
            else "- Perspective summary metrics: skipped"
        ),
        *one_agent_extra,
        plot_status,
    ]

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"build_summary_report: output={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
