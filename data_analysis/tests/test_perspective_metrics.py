from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


class PerspectiveMetricsTest(unittest.TestCase):
    def test_perspective_metrics_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_perspective_raw.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_dir = tmp_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

            config = {
                "raw_glob": str(raw_dir / "*.jsonl"),
                "perspective_question_output": str(tmp_dir / "metrics" / "perspective_question.csv"),
                "perspective_summary_output": str(tmp_dir / "metrics" / "perspective_summary.csv"),
                "run_label": "perspective_test",
                "schema_profile": "multi_agent",
                "hf_dataset_revision": "a70d430",
            }
            config_path = tmp_dir / "config.yaml"
            config_path.write_text(
                "\n".join([f"{k}: {v}" for k, v in config.items()]) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "python3",
                    "scripts/compute_perspective_metrics.py",
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root / "data_analysis",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            question_rows = self._read_csv(Path(config["perspective_question_output"]))
            self.assertEqual(len(question_rows), 2)

            row1 = {k: v for k, v in question_rows[0].items()}
            row2 = {k: v for k, v in question_rows[1].items()}

            self.assertEqual(row1["delta_volatility"], "1")
            self.assertEqual(row1["delta_stubborn"], "-1")
            self.assertEqual(row1["delta_agreeable"], "1")

            self.assertEqual(row2["delta_volatility"], "-1")
            self.assertEqual(row2["delta_stubborn"], "1")
            self.assertEqual(row2["delta_agreeable"], "-1")

            summary_rows = self._read_csv(Path(config["perspective_summary_output"]))
            self.assertEqual(len(summary_rows), 1)
            summary = summary_rows[0]
            self.assertEqual(summary["n_questions"], "2")
            self.assertEqual(summary["mean_delta_volatility"], "0.0")
            self.assertEqual(summary["mean_delta_stubborn"], "0.0")
            self.assertEqual(summary["mean_delta_agreeable"], "0.0")

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
