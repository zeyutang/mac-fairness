"""Compute legacy-style one-agent condition comparisons.

Purpose:
- Produce compatibility tables similar to earlier analysis scripts, but based
  on current one-agent metric outputs.

Input:
- `one_agent_metrics.csv` (condition-level rows by model/subcategory/format)

Outputs:
- `metrics/{run_label}/legacy_analysis/*_summary.csv`
- `metrics/{run_label}/legacy_analysis/*_win_rates.csv`
- `metrics/{run_label}/legacy_analysis/*_pairwise_deltas.csv`

Notes:
- Runs only when `schema_profile=one_agent`.
- Comparisons are computed per model (not pooled across models).
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import cfg_path, ensure_parent, load_config, parse_args


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summarize_condition(rows: list[dict[str, str]], metric_col: str, condition_col: str) -> list[dict[str, Any]]:
    # social_group = benchmark_subcategory from one-agent detailed table
    strata_col = "choice_display_format" if condition_col == "json_field_order" else "json_field_order"

    per_stratum: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["benchmark_subcategory"], r[condition_col], r[strata_col])
        per_stratum[key].append(float(r[metric_col]))

    per_group_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (model, social_group, condition, _stratum), vals in per_stratum.items():
        per_group_condition[(model, social_group, condition)].append(_mean(vals))

    by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (model, social_group, condition), vals in per_group_condition.items():
        by_condition[(model, condition)].append(_mean(vals))

    out = []
    for model, condition in sorted(by_condition.keys()):
        vals = by_condition[(model, condition)]
        out.append(
            {
                "model": model,
                condition_col: condition,
                "group_score_mean": round(_mean(vals), 6),
                "group_score_median": round(_median(vals), 6),
                "n_social_groups": len(vals),
            }
        )
    return out


def _win_rates(rows: list[dict[str, str]], metric_col: str, condition_col: str, higher_is_better: bool) -> list[dict[str, Any]]:
    by_social_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_social_condition[(r["model"], r["benchmark_subcategory"], r[condition_col])].append(float(r[metric_col]))

    by_social: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (model, social, condition), vals in by_social_condition.items():
        by_social[model][social][condition] = _mean(vals)

    out = []
    for model in sorted(by_social.keys()):
        wins: dict[str, float] = defaultdict(float)
        n_groups = 0
        for _social, cond_map in by_social[model].items():
            if not cond_map:
                continue
            n_groups += 1
            best = max(cond_map.values()) if higher_is_better else min(cond_map.values())
            winners = [k for k, v in cond_map.items() if v == best]
            share = 1.0 / len(winners)
            for w in winners:
                wins[w] += share

        for cond in sorted(wins.keys(), key=lambda k: wins[k], reverse=True):
            out.append(
                {
                    "model": model,
                    condition_col: cond,
                    "wins_weighted": round(wins[cond], 6),
                    "n_groups_compared": n_groups,
                    "win_rate": round(wins[cond] / n_groups, 6) if n_groups else 0.0,
                }
            )
    return out


def _pairwise_deltas(rows: list[dict[str, str]], metric_col: str, condition_col: str) -> list[dict[str, Any]]:
    by_social_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_social_condition[(r["model"], r["benchmark_subcategory"], r[condition_col])].append(float(r[metric_col]))

    by_social: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (model, social, condition), vals in by_social_condition.items():
        by_social[model][social][condition] = _mean(vals)

    all_conds = sorted({r[condition_col] for r in rows})
    out = []
    for model in sorted(by_social.keys()):
        for a, b in itertools.combinations(all_conds, 2):
            deltas: list[float] = []
            for _social, cond_map in by_social[model].items():
                if a in cond_map and b in cond_map:
                    deltas.append(cond_map[a] - cond_map[b])
            if not deltas:
                continue
            out.append(
                {
                    "model": model,
                    "condition_a": a,
                    "condition_b": b,
                    "delta_a_minus_b_mean": round(_mean(deltas), 6),
                    "delta_a_minus_b_median": round(_median(deltas), 6),
                    "n_social_groups_paired": len(deltas),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = ensure_parent(str(path))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()

    out_dir = cfg_path(config, "one_agent_legacy_analysis_dir", "metrics/{run_label}/legacy_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if schema_profile != "one_agent":
        print("analyze_legacy_compat: skipped (schema_profile is not one_agent)")
        return 0

    input_path = cfg_path(config, "one_agent_metrics_output", "metrics/{run_label}/one_agent_metrics.csv")
    if not input_path.exists():
        print("analyze_legacy_compat: skipped (one-agent metrics file missing)")
        return 0

    rows = _load_rows(input_path)
    if not rows:
        print("analyze_legacy_compat: skipped (no rows)")
        return 0

    metrics = [
        ("accuracy_rate", True),
        ("ambig_accuracy", True),
        ("ambig_bias_score", False),
        ("disambig_bias_score", False),
    ]
    condition_cols = ["json_field_order", "choice_display_format"]

    metrics_version = str(config.get("metrics_version", "v1"))

    for metric_col, higher_is_better in metrics:
        for condition_col in condition_cols:
            summary_rows = _summarize_condition(rows, metric_col, condition_col)
            win_rows = _win_rates(rows, metric_col, condition_col, higher_is_better)
            pair_rows = _pairwise_deltas(rows, metric_col, condition_col)

            for r in summary_rows:
                r["metrics_version"] = metrics_version
                r["metric"] = metric_col
            for r in win_rows:
                r["metrics_version"] = metrics_version
                r["metric"] = metric_col
            for r in pair_rows:
                r["metrics_version"] = metrics_version
                r["metric"] = metric_col

            _write_csv(
                out_dir / f"{metric_col}_{condition_col}_summary.csv",
                summary_rows,
                ["metrics_version", "metric", "model", condition_col, "group_score_mean", "group_score_median", "n_social_groups"],
            )
            _write_csv(
                out_dir / f"{metric_col}_{condition_col}_win_rates.csv",
                win_rows,
                ["metrics_version", "metric", "model", condition_col, "wins_weighted", "n_groups_compared", "win_rate"],
            )
            _write_csv(
                out_dir / f"{metric_col}_{condition_col}_pairwise_deltas.csv",
                pair_rows,
                ["metrics_version", "metric", "model", "condition_a", "condition_b", "delta_a_minus_b_mean", "delta_a_minus_b_median", "n_social_groups_paired"],
            )

    print(f"analyze_legacy_compat: output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
