"""Shared helpers used by all data_analysis scripts.

This module centralizes:
- config loading (`--config` YAML/JSON),
- path templating (`{run_label}`, `{hf_dataset_revision}`),
- small type conversions (`to_bool`),
- output path helpers (`ensure_parent`).

All pipeline scripts import from here to keep argument parsing and config
behavior consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/analysis_config.yaml",
        help="Path to analysis config (simple YAML key:value or JSON).",
    )
    return parser.parse_args()


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        data[key] = value
    return data


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _parse_simple_yaml(text)


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def resolve_path_template(path_template: str, config: dict[str, Any]) -> str:
    values = {
        "run_label": str(config.get("run_label", "default")).strip() or "default",
        "schema_profile": str(config.get("schema_profile", "auto")).strip() or "auto",
        "hf_dataset_revision": str(config.get("hf_dataset_revision", "")).strip() or "unknown_revision",
    }
    return str(path_template).format(**values)


def cfg_path(config: dict[str, Any], key: str, default: str) -> Path:
    return Path(resolve_path_template(str(config.get(key, default)), config))


def write_simple_yaml(path: Path, data: dict[str, Any], key_order: list[str] | None = None) -> None:
    if key_order:
        # Keep requested order first, then append any new/unlisted keys so we
        # never silently drop config fields when the schema evolves.
        keys = [k for k in key_order if k in data]
        extras = sorted(k for k in data.keys() if k not in set(keys))
        keys.extend(extras)
    else:
        keys = sorted(data.keys())
    lines: list[str] = []
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        else:
            val_str = str(value)
        if any(ch in val_str for ch in [":", "#", "*", "{", "}", "[", "]", "\"", "'"]):
            val_str = f'"{val_str}"'
        lines.append(f"{key}: {val_str}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
