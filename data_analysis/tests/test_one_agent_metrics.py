from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


class OneAgentMetricsTest(unittest.TestCase):
    def test_one_agent_metrics_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_hf_one_agent.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_dir = tmp_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

            config = {
                "raw_glob": str(raw_dir / "*.jsonl"),
                "schema_profile": "one_agent",
                "run_label": "one_agent_metrics_test",
                "one_agent_metrics_output": str(tmp_dir / "metrics" / "one_agent_metrics.csv"),
                "one_agent_model_subcategory_output": str(
                    tmp_dir / "metrics" / "one_agent_model_subcategory_scores.csv"
                ),
                "one_agent_summary_output": str(tmp_dir / "metrics" / "one_agent_summary_by_model.csv"),
            }
            config_path = tmp_dir / "config.yaml"
            config_path.write_text(
                "\n".join([f"{k}: {v}" for k, v in config.items()]) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                ["python3", "scripts/compute_one_agent_metrics.py", "--config", str(config_path)],
                cwd=repo_root / "data_analysis",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            summary_rows = self._read_csv(Path(config["one_agent_summary_output"]))
            self.assertEqual(len(summary_rows), 1)
            row = summary_rows[0]
            self.assertEqual(row["model"], "qwen3-30b")
            self.assertEqual(row["n_rows"], "2")
            self.assertEqual(row["accuracy_rate"], "0.5")

            model_subcat_rows = self._read_csv(Path(config["one_agent_model_subcategory_output"]))
            self.assertEqual(len(model_subcat_rows), 1)
            subcat_row = model_subcat_rows[0]
            self.assertEqual(subcat_row["model"], "qwen3-30b")
            self.assertEqual(subcat_row["benchmark_subcategory"], "bbq_age_sampled")
            self.assertEqual(subcat_row["n_rows"], "2")

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
