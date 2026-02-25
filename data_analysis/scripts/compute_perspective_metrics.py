"""Compute identity-vs-vanilla conversational behavior metrics.

Input:
- multi-agent raw rows containing `identity_agent_r*` and `vanilla_agent_r*`

Outputs:
- `perspective_question.csv` (question-level volatility/stubborn/agreeable)
- `perspective_summary.csv` (aggregated deltas by grouping keys)

Notes:
- This stage captures perspective asymmetry diagnostics.
- Skips cleanly when data/profile are not multi-agent compatible.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import cfg_path, ensure_parent, load_config, parse_args, resolve_path_template


ROUND_RE = re.compile(r"^(identity_agent|vanilla_agent)_r(\d+)$")


def _extract_sequence(row: dict[str, Any], agent_prefix: str) -> list[str]:
    round_vals: list[tuple[int, str]] = []
    for key, value in row.items():
        match = ROUND_RE.match(key)
        if not match:
            continue
        prefix, round_idx = match.group(1), int(match.group(2))
        if prefix != agent_prefix or value is None:
            continue
        round_vals.append((round_idx, str(value).strip()))

    round_vals.sort(key=lambda x: x[0])
    seq = [val for _idx, val in round_vals]

    final_key = f"{agent_prefix}_final"
    if row.get(final_key) is not None:
        final_val = str(row[final_key]).strip()
        if not seq or final_val != seq[-1]:
            seq.append(final_val)

    return seq


def _volatility(seq: list[str]) -> int:
    if len(seq) < 2:
        return 0
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def _stubborn(own_seq: list[str], other_seq: list[str]) -> int:
    if not own_seq:
        return 0
    base = own_seq[0]
    never_shifted = all(x == base for x in own_seq)
    other_disagreed = any(x != base for x in other_seq)
    return 1 if (never_shifted and other_disagreed) else 0


def _agreeable(own_seq: list[str], other_seq: list[str]) -> int:
    max_steps = min(len(own_seq), len(other_seq))
    if max_steps < 2:
        return 0
    for r in range(1, max_steps):
        shifted = own_seq[r] != own_seq[r - 1]
        matched_other_prev = own_seq[r] == other_seq[r - 1]
        if shifted and matched_other_prev:
            return 1
    return 0


def _safe_str(row: dict[str, Any], key: str, default: str = "") -> str:
    val = row.get(key)
    if val is None:
        return default
    return str(val)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()
    metrics_version = str(config.get("metrics_version", "v1"))

    raw_glob = resolve_path_template(str(config.get("raw_glob", "raw/{hf_dataset_revision}/{run_label}/*.jsonl")), config)
    question_out = cfg_path(config, "perspective_question_output", "metrics/{run_label}/perspective_question.csv")
    summary_out = cfg_path(config, "perspective_summary_output", "metrics/{run_label}/perspective_summary.csv")

    if schema_profile == "one_agent":
        if question_out.exists():
            question_out.unlink()
        if summary_out.exists():
            summary_out.unlink()
        print("compute_perspective_metrics: skipped for schema_profile=one_agent")
        return 0

    question_out = ensure_parent(str(question_out))
    summary_out = ensure_parent(str(summary_out))

    rows_out: list[dict[str, Any]] = []
    files = sorted(glob.glob(raw_glob, recursive=True))

    for path_str in files:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                id_seq = _extract_sequence(row, "identity_agent")
                van_seq = _extract_sequence(row, "vanilla_agent")
                if not id_seq or not van_seq:
                    continue

                id_vol = _volatility(id_seq)
                van_vol = _volatility(van_seq)
                id_stub = _stubborn(id_seq, van_seq)
                van_stub = _stubborn(van_seq, id_seq)
                id_agree = _agreeable(id_seq, van_seq)
                van_agree = _agreeable(van_seq, id_seq)

                rows_out.append(
                    {
                        "metrics_version": metrics_version,
                        "transcript_id": _safe_str(row, "transcript_id"),
                        "question_id": _safe_str(row, "question_id"),
                        "model": _safe_str(row, "model"),
                        "benchmark_subcategory": _safe_str(row, "benchmark_subcategory"),
                        "experiment_name": _safe_str(row, "experiment_name"),
                        "persona": _safe_str(row, "persona"),
                        "demographics": _safe_str(row, "demographics"),
                        "if_as_human": _safe_str(row, "if_as_human"),
                        "reveal_condition": _safe_str(row, "reveal_condition"),
                        "identity_volatility": id_vol,
                        "vanilla_volatility": van_vol,
                        "identity_stubborn": id_stub,
                        "vanilla_stubborn": van_stub,
                        "identity_agreeable": id_agree,
                        "vanilla_agreeable": van_agree,
                        "delta_volatility": id_vol - van_vol,
                        "delta_stubborn": id_stub - van_stub,
                        "delta_agreeable": id_agree - van_agree,
                    }
                )

    q_fields = [
        "metrics_version",
        "transcript_id",
        "question_id",
        "model",
        "benchmark_subcategory",
        "experiment_name",
        "persona",
        "demographics",
        "if_as_human",
        "reveal_condition",
        "identity_volatility",
        "vanilla_volatility",
        "identity_stubborn",
        "vanilla_stubborn",
        "identity_agreeable",
        "vanilla_agreeable",
        "delta_volatility",
        "delta_stubborn",
        "delta_agreeable",
    ]
    with question_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=q_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows_out:
        key = (
            str(row["benchmark_subcategory"]),
            str(row["reveal_condition"]),
            str(row["demographics"]),
            str(row["if_as_human"]),
        )
        grouped[key]["n_questions"] += 1
        for metric in (
            "identity_volatility",
            "vanilla_volatility",
            "identity_stubborn",
            "vanilla_stubborn",
            "identity_agreeable",
            "vanilla_agreeable",
            "delta_volatility",
            "delta_stubborn",
            "delta_agreeable",
        ):
            grouped[key][metric] += float(row[metric])

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        bmk, reveal, demo, human = key
        bucket = grouped[key]
        n = int(bucket["n_questions"])
        out: dict[str, Any] = {
            "metrics_version": metrics_version,
            "benchmark_subcategory": bmk,
            "reveal_condition": reveal,
            "demographics": demo,
            "if_as_human": human,
            "n_questions": n,
        }
        for metric in (
            "identity_volatility",
            "vanilla_volatility",
            "identity_stubborn",
            "vanilla_stubborn",
            "identity_agreeable",
            "vanilla_agreeable",
            "delta_volatility",
            "delta_stubborn",
            "delta_agreeable",
        ):
            out[f"mean_{metric}"] = round(bucket[metric] / n, 6) if n else 0.0
        summary_rows.append(out)

    s_fields = [
        "metrics_version",
        "benchmark_subcategory",
        "reveal_condition",
        "demographics",
        "if_as_human",
        "n_questions",
        "mean_identity_volatility",
        "mean_vanilla_volatility",
        "mean_identity_stubborn",
        "mean_vanilla_stubborn",
        "mean_identity_agreeable",
        "mean_vanilla_agreeable",
        "mean_delta_volatility",
        "mean_delta_stubborn",
        "mean_delta_agreeable",
    ]
    with summary_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=s_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"compute_perspective_metrics: files={len(files)} question_rows={len(rows_out)} question_out={question_out} summary_out={summary_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
