"""Download JSONL files from a Hugging Face dataset repo into the active batch.

Inputs:
- active config (`hf_dataset_repo`, `hf_dataset_revision`, `raw_glob`)
- optional `--include` glob override

Behavior:
- resolves target raw directory from config,
- runs `hf download` (or `huggingface-cli download`) command,
- writes downloaded files into batch-specific raw folder.

This is called by `make ingest_hf` after `prepare_batch`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

from _common import load_config, resolve_path_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HF dataset JSONL files into the prepared batch folder.")
    parser.add_argument("--config", default="configs/analysis_config.yaml")
    parser.add_argument("--include", default="", help="Glob pattern for files to download (default from config or *.jsonl).")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    return parser.parse_args()


def _repo_id(repo_value: str) -> str:
    value = repo_value.strip().rstrip("/")
    if not value:
        return ""
    marker = "/datasets/"
    if marker in value:
        return value.split(marker, 1)[1]
    return value


def _raw_dir_from_glob(config: dict[str, Any]) -> Path:
    raw_glob = resolve_path_template(str(config.get("raw_glob", "raw/{hf_dataset_revision}/{run_label}/*.jsonl")), config)
    wildcard_positions = [pos for ch in ("*", "?", "[") for pos in [raw_glob.find(ch)] if pos != -1]
    if wildcard_positions:
        cut = min(wildcard_positions)
        prefix = raw_glob[:cut]
        if prefix.endswith("/"):
            return Path(prefix.rstrip("/"))
        return Path(prefix).parent
    path = Path(raw_glob)
    return path.parent if path.suffix else path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    repo = _repo_id(str(config.get("hf_dataset_repo", "")))
    revision = str(config.get("hf_dataset_revision", "")).strip()
    include = args.include.strip() or str(config.get("hf_include_glob", "*.jsonl")).strip() or "*.jsonl"
    local_dir = _raw_dir_from_glob(config)

    if not repo:
        raise SystemExit("download_hf_batch: hf_dataset_repo is required in config")
    if not revision:
        raise SystemExit("download_hf_batch: hf_dataset_revision is required in config")

    local_dir.mkdir(parents=True, exist_ok=True)

    cli = shutil.which("huggingface-cli")
    hf = shutil.which("hf")
    if cli:
        cmd = [
            "huggingface-cli",
            "download",
            repo,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--include",
            include,
            "--local-dir",
            str(local_dir),
        ]
    elif hf:
        cmd = [
            "hf",
            "download",
            repo,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--include",
            include,
            "--local-dir",
            str(local_dir),
        ]
    elif args.dry_run:
        cmd = [
            "hf",
            "download",
            repo,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--include",
            include,
            "--local-dir",
            str(local_dir),
        ]
    else:
        raise SystemExit(
            "download_hf_batch: neither huggingface-cli nor hf command is available. Install huggingface_hub."
        )

    print("download_hf_batch: local_dir=" + str(local_dir))
    print("download_hf_batch: command=" + " ".join(cmd))

    if args.dry_run:
        return 0

    subprocess.run(cmd, check=True)
    downloaded_jsonl = list(local_dir.rglob("*.jsonl"))
    if not downloaded_jsonl:
        raise SystemExit(
            "download_hf_batch: download completed but no .jsonl files were found in local-dir. "
            "Check hf_include_glob/HF_INCLUDE (example: sampled-set-with-rep-exps/bbq-sample/*.jsonl)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
