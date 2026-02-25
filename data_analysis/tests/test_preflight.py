from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


class PreflightTest(unittest.TestCase):
    def test_preflight_passes_for_matching_schema(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_hf_one_agent.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_dir = tmp_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            cfg = tmp_dir / "cfg.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        f"raw_glob: {raw_dir}/*.jsonl",
                        "schema_profile: one_agent",
                        "run_label: test",
                        "hf_dataset_revision: main",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                ["python3", "scripts/preflight_validate.py", "--config", str(cfg)],
                cwd=repo_root / "data_analysis",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)

    def test_preflight_fails_when_no_files_match(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg = tmp_dir / "cfg.yaml"
            cfg.write_text("raw_glob: does/not/exist/*.jsonl\n", encoding="utf-8")
            proc = subprocess.run(
                ["python3", "scripts/preflight_validate.py", "--config", str(cfg)],
                cwd=repo_root / "data_analysis",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
