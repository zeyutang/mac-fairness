"""Aggregate behavior metrics to model-level summaries.

Input:
- `metrics/{run_label}/behavior_metrics.csv` from `compute_behavior_metrics.py`

Output:
- `metrics/{run_label}/behavior_metrics_by_model.csv`

Notes:
- Skips one-agent runs because those use one-agent specific outputs.
"""

from __future__ import annotations

import csv
from collections import defaultdict

from _common import cfg_path, ensure_parent, load_config, parse_args


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()
    metrics_version = str(config.get("metrics_version", "v1"))

    metrics_path = cfg_path(config, "metrics_output", "metrics/{run_label}/behavior_metrics.csv")
    output_path = cfg_path(config, "aggregate_output", "metrics/{run_label}/behavior_metrics_by_model.csv")

    if schema_profile == "one_agent":
        if output_path.exists():
            output_path.unlink()
        print("aggregate_metrics: skipped for schema_profile=one_agent")
        return 0
    if not metrics_path.exists():
        if output_path.exists():
            output_path.unlink()
        print("aggregate_metrics: skipped because behavior metrics file is missing")
        return 0

    output_path = ensure_parent(str(output_path))

    by_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            counts[model] += 1
            for key in (
                "changed_after_wrong_rate",
                "stood_firm_wrong_rate",
                "corrected_after_wrong_rate",
            ):
                by_model[model][key] += float(row[key])
            by_model[model]["initial_coverage_rate"] += float(row["total_with_initial_answer"]) / max(
                float(row["total_rows"]), 1.0
            )

    fieldnames = [
        "metrics_version",
        "model",
        "n_runs",
        "avg_changed_after_wrong_rate",
        "avg_stood_firm_wrong_rate",
        "avg_corrected_after_wrong_rate",
        "avg_initial_coverage_rate",
    ]

    rows = []
    for model in sorted(by_model.keys()):
        n = counts[model]
        rows.append(
            {
                "metrics_version": metrics_version,
                "model": model,
                "n_runs": n,
                "avg_changed_after_wrong_rate": round(by_model[model]["changed_after_wrong_rate"] / n, 6),
                "avg_stood_firm_wrong_rate": round(by_model[model]["stood_firm_wrong_rate"] / n, 6),
                "avg_corrected_after_wrong_rate": round(by_model[model]["corrected_after_wrong_rate"] / n, 6),
                "avg_initial_coverage_rate": round(by_model[model]["initial_coverage_rate"] / n, 6),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"aggregate_metrics: models={len(rows)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
