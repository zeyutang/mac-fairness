"""Compute core change/correction behavior metrics from normalized rows.

Input:
- `intermediate/{run_label}/normalized.jsonl`

Output:
- `metrics/{run_label}/behavior_metrics.csv`

Metrics include counts/rates such as:
- changed after wrong,
- stood firm while wrong,
- corrected after wrong.

Notes:
- For one-agent runs, normalization marks rows without initial answers, so
  change-based denominators are handled safely.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from _common import cfg_path, ensure_parent, load_config, parse_args, to_bool


GROUP_KEYS = ["run_id", "model"]


class Counter:
    def __init__(self) -> None:
        self.total_rows = 0
        self.total_with_initial_answer = 0
        self.total_without_initial_answer = 0
        self.total_initially_wrong = 0
        self.changed_after_wrong_count = 0
        self.stood_firm_wrong_count = 0
        self.corrected_after_wrong_count = 0
        self.incorrect_after_change_count = 0

    def consume(self, row: dict[str, object]) -> None:
        self.total_rows += 1
        has_initial = to_bool(row.get("has_initial_answer"))
        if has_initial is None:
            has_initial = True

        final_is_correct = to_bool(row.get("final_is_correct"))
        changed_answer = to_bool(row.get("changed_answer"))
        initial_is_correct = to_bool(row.get("initial_is_correct"))

        if final_is_correct is None or changed_answer is None:
            return

        if not has_initial:
            self.total_without_initial_answer += 1
            return
        self.total_with_initial_answer += 1
        if initial_is_correct is None:
            return

        if not initial_is_correct:
            self.total_initially_wrong += 1
            if changed_answer:
                self.changed_after_wrong_count += 1
            if (not changed_answer) and (not final_is_correct):
                self.stood_firm_wrong_count += 1
            if final_is_correct:
                self.corrected_after_wrong_count += 1

        if changed_answer and (not final_is_correct):
            self.incorrect_after_change_count += 1

    def to_row(self, run_id: str, model: str, metrics_version: str) -> dict[str, object]:
        denom = self.total_initially_wrong

        def safe_rate(numerator: int) -> float:
            if denom == 0:
                return 0.0
            return numerator / denom

        return {
            "metrics_version": metrics_version,
            "run_id": run_id,
            "model": model,
            "total_rows": self.total_rows,
            "total_with_initial_answer": self.total_with_initial_answer,
            "total_without_initial_answer": self.total_without_initial_answer,
            "total_initially_wrong": self.total_initially_wrong,
            "changed_after_wrong_count": self.changed_after_wrong_count,
            "stood_firm_wrong_count": self.stood_firm_wrong_count,
            "corrected_after_wrong_count": self.corrected_after_wrong_count,
            "incorrect_after_change_count": self.incorrect_after_change_count,
            "changed_after_wrong_rate": round(safe_rate(self.changed_after_wrong_count), 6),
            "stood_firm_wrong_rate": round(safe_rate(self.stood_firm_wrong_count), 6),
            "corrected_after_wrong_rate": round(safe_rate(self.corrected_after_wrong_count), 6),
        }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()

    input_path = cfg_path(config, "normalized_output", "intermediate/{run_label}/normalized.jsonl")
    output_path = cfg_path(config, "metrics_output", "metrics/{run_label}/behavior_metrics.csv")

    if schema_profile == "one_agent":
        if output_path.exists():
            output_path.unlink()
        print("compute_behavior_metrics: skipped for schema_profile=one_agent (requires initial+final answers)")
        return 0

    output_path = ensure_parent(str(output_path))

    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = tuple(str(row[k]) for k in GROUP_KEYS)
            groups[key].consume(row)

    metrics_version = str(config.get("metrics_version", "v1"))
    rows = [groups[key].to_row(key[0], key[1], metrics_version) for key in sorted(groups.keys())]

    fieldnames = [
        "metrics_version",
        "run_id",
        "model",
        "total_rows",
        "total_with_initial_answer",
        "total_without_initial_answer",
        "total_initially_wrong",
        "changed_after_wrong_count",
        "stood_firm_wrong_count",
        "corrected_after_wrong_count",
        "incorrect_after_change_count",
        "changed_after_wrong_rate",
        "stood_firm_wrong_rate",
        "corrected_after_wrong_rate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"compute_behavior_metrics: groups={len(rows)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
