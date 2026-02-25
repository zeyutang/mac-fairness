"""Validate raw inputs before running analysis stages.

Checks:
- raw_glob resolves to at least one JSONL file,
- sample rows parse as JSON,
- inferred schema profile matches configured profile (or warns/fails).

This stage is intentionally early in `make all` to fail fast on bad inputs.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from _common import load_config, parse_args, resolve_path_template


def _detect_profile(row: dict[str, Any]) -> str:
    if any(k in row for k in ("vanilla_agent_r0", "identity_agent_r0")):
        return "multi_agent"
    if any(k in row for k in ("final_answer", "is_correct", "opinion")):
        return "one_agent"
    return "generic"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    schema_profile = str(config.get("schema_profile", "auto")).strip().lower()
    raw_glob = resolve_path_template(str(config.get("raw_glob", "raw/{hf_dataset_revision}/{run_label}/**/*.jsonl")), config)
    files = sorted(glob.glob(raw_glob, recursive=True))

    if not files:
        print(f"preflight_validate: FAIL no files matched raw_glob={raw_glob}")
        return 2

    sample_rows: list[dict[str, Any]] = []
    for fp in files:
        path = Path(fp)
        if not path.name.endswith(".jsonl"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(sample_rows) >= 200:
                    break
        if len(sample_rows) >= 200:
            break

    if not sample_rows:
        print("preflight_validate: FAIL matched files exist but no readable JSONL rows")
        return 2

    profile_counts = {"one_agent": 0, "multi_agent": 0, "generic": 0}
    for row in sample_rows:
        profile_counts[_detect_profile(row)] += 1

    dominant_profile = max(profile_counts, key=profile_counts.get)
    dominant_ratio = profile_counts[dominant_profile] / max(len(sample_rows), 1)

    if schema_profile != "auto":
        expected = schema_profile
        match_ratio = profile_counts.get(expected, 0) / max(len(sample_rows), 1)
        if match_ratio < 0.6:
            print(
                "preflight_validate: FAIL schema_profile mismatch "
                f"expected={expected} match_ratio={match_ratio:.3f} detected={profile_counts}"
            )
            return 2

    total_bytes = sum(Path(f).stat().st_size for f in files if Path(f).is_file())
    print(
        "preflight_validate: OK "
        f"files={len(files)} sample_rows={len(sample_rows)} dominant_profile={dominant_profile} "
        f"dominant_ratio={dominant_ratio:.3f} total_bytes={total_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
