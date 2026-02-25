"""Write reproducibility metadata for the current run.

Outputs:
- `run_metadata.json` with config/runtime context,
- `raw_file_manifest.json` with file paths, sizes, and SHA256 checksums.

This provides traceability for exactly which raw files produced a report.
"""

from __future__ import annotations

import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from _common import cfg_path, ensure_parent, load_config, parse_args, resolve_path_template


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    output_path = ensure_parent(str(cfg_path(config, "run_metadata_output", "reports/{run_label}/run_metadata.json")))
    manifest_path = ensure_parent(str(cfg_path(config, "raw_manifest_output", "reports/{run_label}/raw_file_manifest.json")))

    raw_glob = resolve_path_template(str(config.get("raw_glob", "raw/{hf_dataset_revision}/{run_label}/**/*.jsonl")), config)
    raw_files = sorted(Path(p) for p in glob.glob(raw_glob, recursive=True) if Path(p).is_file())

    raw_manifest = []
    total_bytes = 0
    for p in raw_files:
        size = p.stat().st_size
        total_bytes += size
        raw_manifest.append(
            {
                "path": str(p),
                "size_bytes": size,
                "sha256": _sha256(p),
            }
        )

    manifest_path.write_text(json.dumps(raw_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_version": config.get("metrics_version", "v1"),
        "hf_dataset_repo": config.get("hf_dataset_repo", ""),
        "hf_dataset_revision": config.get("hf_dataset_revision", ""),
        "hf_snapshot_date": config.get("hf_snapshot_date", ""),
        "run_label": config.get("run_label", "default"),
        "schema_profile": config.get("schema_profile", "auto"),
        "raw_glob": raw_glob,
        "raw_files_count": len(raw_files),
        "raw_total_bytes": total_bytes,
        "raw_manifest_path": str(manifest_path),
        "normalized_output": str(cfg_path(config, "normalized_output", "intermediate/{run_label}/normalized.jsonl")),
        "config_path": args.config,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"write_run_metadata: output={output_path} raw_files={len(raw_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
