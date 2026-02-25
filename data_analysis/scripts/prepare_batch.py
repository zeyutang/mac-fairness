"""Create/run-select a batch config and folder layout for a new analysis pull.

Responsibilities:
- infer schema profile from a sample JSONL row (optional),
- generate/normalize `run_label`,
- write immutable run config to `configs/runs/<run_label>.yaml`,
- update active config pointer (`configs/.active_run_config`),
- ensure batch raw directory structure exists.

This is the entry point for reproducible batch isolation.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import load_config, write_simple_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare analysis config and raw folder for a new HF pull batch.")
    parser.add_argument("--config", default="configs/analysis_config.yaml", help="Base config template path.")
    parser.add_argument("--sample-jsonl", default="", help="Optional sample JSONL file for metadata inference.")
    parser.add_argument("--hf-revision", default="", help="HF dataset revision (commit hash preferred).")
    parser.add_argument("--schema-profile", default="", help="Override schema profile: auto|one_agent|multi_agent|generic")
    parser.add_argument("--run-label", default="", help="Manual run label override.")
    parser.add_argument("--date", default="", help="Date prefix YYYY-MM-DD (defaults to today UTC).")
    parser.add_argument("--raw-root", default="raw", help="Root directory for raw batch folders.")
    parser.add_argument("--run-config-dir", default="configs/runs", help="Directory to write immutable run configs.")
    parser.add_argument("--active-config-file", default="configs/.active_run_config", help="File storing active run config path.")
    parser.add_argument("--in-place", action="store_true", help="Mutate --config directly (legacy behavior).")
    return parser.parse_args()


def _read_first_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    return None


def _infer_schema_profile(sample: dict[str, Any] | None, configured: str) -> str:
    if configured and configured != "auto":
        return configured
    if not sample:
        return "auto"
    if any(k in sample for k in ("vanilla_agent_r0", "identity_agent_r0")):
        return "multi_agent"
    if any(k in sample for k in ("final_answer", "is_correct", "opinion")):
        return "one_agent"
    return "generic"


def _infer_benchmark(sample: dict[str, Any] | None) -> str:
    if sample:
        subcat = sample.get("benchmark_subcategory")
        if isinstance(subcat, str) and subcat.strip():
            bench = subcat.strip().lower().replace("_sampled", "")
            return _sanitize_token(bench)

        qid = sample.get("question_id")
        if isinstance(qid, str) and "_" in qid:
            parts = qid.split("_")
            if len(parts) >= 2:
                return _sanitize_token("_".join(parts[:-1]))

    return "unknown_benchmark"


def _infer_agent_mode(schema_profile: str, sample: dict[str, Any] | None) -> str:
    if schema_profile == "one_agent":
        return "1agent"
    if schema_profile == "multi_agent":
        if sample and isinstance(sample.get("experiment_name"), str):
            exp = sample["experiment_name"]
            match = re.search(r"(\d+agent)", exp)
            if match:
                return match.group(1)
        return "multiagent"
    if sample and isinstance(sample.get("experiment_name"), str):
        exp = sample["experiment_name"]
        match = re.search(r"(\d+agent)", exp)
        if match:
            return match.group(1)
    return "generic"


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main() -> int:
    args = parse_args()
    base_config_path = Path(args.config)
    config = load_config(args.config)

    sample_row = _read_first_jsonl_row(Path(args.sample_jsonl)) if args.sample_jsonl else None

    hf_revision = (args.hf_revision or str(config.get("hf_dataset_revision", ""))).strip()
    if not hf_revision:
        hf_revision = "unknown_revision"

    configured_profile = (args.schema_profile or str(config.get("schema_profile", "auto"))).strip() or "auto"
    schema_profile = _infer_schema_profile(sample_row, configured_profile)

    if args.run_label:
        run_label = _sanitize_token(args.run_label)
    else:
        date_prefix = args.date.strip() if args.date else _today_utc()
        if sample_row is None:
            run_label = _sanitize_token(f"{date_prefix}_hfpull_{hf_revision[:7]}")
        else:
            benchmark = _infer_benchmark(sample_row)
            agent_mode = _infer_agent_mode(schema_profile, sample_row)
            run_label = _sanitize_token(f"{date_prefix}_{benchmark}_{agent_mode}")

    config["run_label"] = run_label
    config["schema_profile"] = schema_profile
    config["hf_dataset_revision"] = hf_revision
    config.setdefault("raw_glob", "raw/{hf_dataset_revision}/{run_label}/**/*.jsonl")
    config.setdefault("metrics_version", "v1")

    raw_root = Path(args.raw_root)
    raw_dir = raw_root / hf_revision / run_label
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.in_place:
        out_config_path = base_config_path
    else:
        run_config_dir = Path(args.run_config_dir)
        run_config_dir.mkdir(parents=True, exist_ok=True)
        out_config_path = run_config_dir / f"{run_label}.yaml"

    write_simple_yaml(
        out_config_path,
        config,
        key_order=[
            "run_label",
            "schema_profile",
            "raw_glob",
            "normalized_output",
            "run_metadata_output",
            "metrics_output",
            "aggregate_output",
            "perspective_question_output",
            "perspective_summary_output",
            "one_agent_metrics_output",
            "one_agent_model_subcategory_output",
            "one_agent_summary_output",
            "one_agent_legacy_analysis_dir",
            "report_output",
            "plots_dir",
            "raw_manifest_output",
            "metrics_version",
            "hf_dataset_repo",
            "hf_dataset_revision",
            "hf_snapshot_date",
            "hf_include_glob",
        ],
    )

    if args.active_config_file:
        active_path = Path(args.active_config_file)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(str(out_config_path) + "\n", encoding="utf-8")

    print(f"prepare_batch: run_label={run_label}")
    print(f"prepare_batch: schema_profile={schema_profile}")
    print(f"prepare_batch: hf_dataset_revision={hf_revision}")
    print(f"prepare_batch: raw_dir={raw_dir}")
    print(f"prepare_batch: run_config={out_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
