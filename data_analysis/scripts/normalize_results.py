"""Normalize heterogeneous raw JSONL rows into a canonical schema.

Input:
- raw files matched by `raw_glob`

Output:
- `intermediate/{run_label}/normalized.jsonl`

Supports:
- generic single-record schemas,
- one-agent rows (`final_answer`, `is_correct`, `opinion`),
- multi-agent transcript rows (`vanilla_agent_*`, `identity_agent_*`).

The normalized schema is the contract for downstream behavior metric scripts.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Iterable

from _common import cfg_path, ensure_parent, load_config, parse_args, resolve_path_template, to_bool

REQUIRED_FIELDS = [
    "run_id",
    "model",
    "question_id",
    "answer_before",
    "answer_after",
    "correct_answer",
]

FIELD_ALIASES = {
    "run_id": ["run_id", "experiment_name", "run_name"],
    "model": ["model", "model_name", "model_id"],
    "question_id": ["question_id", "item_id", "id"],
    "answer_before": ["answer_before", "initial_answer", "pre_deliberation_answer", "first_answer"],
    "answer_after": ["answer_after", "final_answer", "post_deliberation_answer", "opinion"],
    "correct_answer": ["correct_answer", "correct_answer_id", "gold_answer", "label"],
}


def _extract_value(row: dict[str, Any], canonical_key: str) -> Any:
    for key in FIELD_ALIASES[canonical_key]:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _normalize_generic_row(row: dict[str, Any]) -> dict[str, Any] | None:
    normalized: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        value = _extract_value(row, field)
        if value is None:
            return None
        normalized[field] = str(value).strip()

    before = normalized["answer_before"]
    after = normalized["answer_after"]
    correct = normalized["correct_answer"]

    normalized["changed_answer"] = before != after
    normalized["initial_is_correct"] = before == correct
    normalized["final_is_correct"] = after == correct
    normalized["has_initial_answer"] = True

    if "conversation_id" in row and row["conversation_id"] is not None:
        normalized["conversation_id"] = str(row["conversation_id"])

    return normalized


def _build_multi_agent_row(row: dict[str, Any], agent: str) -> dict[str, Any] | None:
    before_key = f"{agent}_r0"
    after_key = f"{agent}_final"

    if row.get(before_key) is None or row.get(after_key) is None:
        return None

    run_id = row.get("experiment_name") or row.get("run_id")
    model = row.get("model")
    question_id = row.get("question_id")
    correct_answer = row.get("correct_answer")
    if run_id is None or model is None or question_id is None or correct_answer is None:
        return None

    before = str(row[before_key]).strip()
    after = str(row[after_key]).strip()
    correct = str(correct_answer).strip()

    shifted = to_bool(row.get(f"{agent}_shifted"))
    initial_correct = to_bool(row.get(f"{agent}_correct_r0"))
    final_correct = to_bool(row.get(f"{agent}_correct_final"))

    normalized: dict[str, Any] = {
        "run_id": str(run_id).strip(),
        "model": f"{str(model).strip()}:{agent}",
        "analysis_agent": agent,
        "question_id": str(question_id).strip(),
        "answer_before": before,
        "answer_after": after,
        "correct_answer": correct,
        "changed_answer": (before != after) if shifted is None else shifted,
        "initial_is_correct": (before == correct) if initial_correct is None else initial_correct,
        "final_is_correct": (after == correct) if final_correct is None else final_correct,
        "has_initial_answer": True,
    }

    if row.get("transcript_id") is not None:
        normalized["transcript_id"] = str(row["transcript_id"])

    return normalized


def _normalize_multi_agent_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for agent in ("vanilla_agent", "identity_agent"):
        normalized = _build_multi_agent_row(row, agent)
        if normalized is not None:
            out.append(normalized)
    return out


def _normalize_one_agent_row(row: dict[str, Any]) -> dict[str, Any] | None:
    run_id = row.get("experiment_name") or row.get("run_id")
    model = row.get("model")
    question_id = row.get("question_id")
    correct_answer = row.get("correct_answer")

    answer_after = (
        row.get("final_answer")
        or row.get("opinion")
        or row.get("answer_after")
        or row.get("final")
    )
    if run_id is None or model is None or question_id is None or correct_answer is None or answer_after is None:
        return None

    # One-agent rows do not include a pre-deliberation answer.
    answer_after_s = str(answer_after).strip()
    correct_s = str(correct_answer).strip()
    final_correct = to_bool(row.get("is_correct"))
    if final_correct is None:
        final_correct = answer_after_s == correct_s

    normalized: dict[str, Any] = {
        "run_id": str(run_id).strip(),
        "model": str(model).strip(),
        "analysis_agent": "single_agent",
        "question_id": str(question_id).strip(),
        "answer_before": answer_after_s,
        "answer_after": answer_after_s,
        "correct_answer": correct_s,
        "changed_answer": False,
        "initial_is_correct": final_correct,
        "final_is_correct": final_correct,
        "has_initial_answer": False,
    }

    if row.get("transcript_id") is not None:
        normalized["transcript_id"] = str(row["transcript_id"])

    return normalized


def _detect_profile(row: dict[str, Any]) -> str:
    if any(k in row for k in ("vanilla_agent_r0", "identity_agent_r0")):
        return "multi_agent"
    if any(k in row for k in ("final_answer", "is_correct", "opinion")):
        return "one_agent"
    return "generic"


def normalize_row(row: dict[str, Any], schema_profile: str) -> Iterable[dict[str, Any]]:
    profile = schema_profile
    if profile == "auto":
        profile = _detect_profile(row)

    if profile == "multi_agent":
        return _normalize_multi_agent_row(row)
    if profile == "one_agent":
        normalized = _normalize_one_agent_row(row)
        return [] if normalized is None else [normalized]

    generic = _normalize_generic_row(row)
    return [] if generic is None else [generic]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    raw_glob = resolve_path_template(str(config.get("raw_glob", "raw/{hf_dataset_revision}/{run_label}/*.jsonl")), config)
    out_path = ensure_parent(str(cfg_path(config, "normalized_output", "intermediate/{run_label}/normalized.jsonl")))
    schema_profile = str(config.get("schema_profile", "auto")).strip() or "auto"

    input_files = sorted(Path(p) for p in glob.glob(raw_glob, recursive=True))
    normalized_rows: list[dict[str, Any]] = []
    dropped = 0

    for path in input_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                normalized_rows_for_line = list(normalize_row(row, schema_profile=schema_profile))
                if not normalized_rows_for_line:
                    dropped += 1
                    continue
                normalized_rows.extend(normalized_rows_for_line)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in normalized_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        f"normalize_results: files={len(input_files)} kept={len(normalized_rows)} dropped={dropped} profile={schema_profile} output={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
