from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


class PipelineTest(unittest.TestCase):
    def test_end_to_end_pipeline_with_fixture(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_raw.jsonl"
        metrics_rows, agg_rows, config = self._run_pipeline(repo_root, fixture)

        self.assertEqual(len(metrics_rows), 2)
        by_key = {(row["run_id"], row["model"]): row for row in metrics_rows}
        run1 = by_key[("run1", "modelA")]
        self.assertEqual(run1["total_rows"], "4")
        self.assertEqual(run1["total_initially_wrong"], "3")
        self.assertEqual(run1["changed_after_wrong_count"], "2")
        self.assertEqual(run1["stood_firm_wrong_count"], "1")
        self.assertEqual(run1["corrected_after_wrong_count"], "1")
        self.assertEqual(run1["incorrect_after_change_count"], "1")
        self.assertEqual(len(agg_rows), 2)

    def test_end_to_end_pipeline_with_hf_schema_fixture(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_hf_raw.jsonl"
        metrics_rows, _, _config = self._run_pipeline(repo_root, fixture)

        self.assertEqual(len(metrics_rows), 2)
        by_key = {(row["run_id"], row["model"]): row for row in metrics_rows}

        vanilla = by_key[("run_hf", "modelA:vanilla_agent")]
        self.assertEqual(vanilla["total_rows"], "2")
        self.assertEqual(vanilla["total_initially_wrong"], "2")
        self.assertEqual(vanilla["changed_after_wrong_count"], "1")
        self.assertEqual(vanilla["stood_firm_wrong_count"], "1")
        self.assertEqual(vanilla["corrected_after_wrong_count"], "1")

        identity = by_key[("run_hf", "modelA:identity_agent")]
        self.assertEqual(identity["total_rows"], "2")
        self.assertEqual(identity["total_initially_wrong"], "2")
        self.assertEqual(identity["changed_after_wrong_count"], "1")
        self.assertEqual(identity["stood_firm_wrong_count"], "1")
        self.assertEqual(identity["corrected_after_wrong_count"], "1")

    def test_end_to_end_pipeline_with_hf_one_agent_fixture(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_hf_one_agent.jsonl"
        metrics_rows, _, _config = self._run_pipeline(
            repo_root,
            fixture,
            overrides={"schema_profile": "one_agent", "run_label": "one_agent_batch"},
        )

        self.assertEqual(len(metrics_rows), 0)

    def _run_pipeline(
        self,
        repo_root: Path,
        fixture_path: Path,
        overrides: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_dir = tmp_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample_raw.jsonl").write_text(
                fixture_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            config: dict[str, str] = {
                "raw_glob": str(raw_dir / "*.jsonl"),
                "normalized_output": str(tmp_dir / "intermediate" / "normalized.jsonl"),
                "run_metadata_output": str(tmp_dir / "reports" / "run_metadata.json"),
                "metrics_output": str(tmp_dir / "metrics" / "behavior_metrics.csv"),
                "aggregate_output": str(tmp_dir / "metrics" / "behavior_metrics_by_model.csv"),
                "perspective_question_output": str(tmp_dir / "metrics" / "perspective_question.csv"),
                "perspective_summary_output": str(tmp_dir / "metrics" / "perspective_summary.csv"),
                "one_agent_metrics_output": str(tmp_dir / "metrics" / "one_agent_metrics.csv"),
                "one_agent_model_subcategory_output": str(
                    tmp_dir / "metrics" / "one_agent_model_subcategory_scores.csv"
                ),
                "one_agent_summary_output": str(tmp_dir / "metrics" / "one_agent_summary_by_model.csv"),
                "report_output": str(tmp_dir / "reports" / "summary.md"),
                "plots_dir": str(tmp_dir / "plots"),
                "run_label": "test_batch",
                "schema_profile": "auto",
                "hf_dataset_repo": "org/dataset",
                "hf_dataset_revision": "abc123",
                "hf_snapshot_date": "2026-02-24",
            }
            if overrides:
                config.update(overrides)
            config_path = tmp_dir / "config.yaml"
            config_path.write_text(
                "\n".join([f"{k}: {v}" for k, v in config.items()]) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                ["make", "-C", "data_analysis", "all", f"CONFIG={config_path}"],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertTrue(Path(config["report_output"]).exists())

            is_one_agent = config.get("schema_profile") == "one_agent"
            if is_one_agent:
                self.assertFalse(Path(config["metrics_output"]).exists())
                self.assertFalse(Path(config["aggregate_output"]).exists())
                self.assertFalse(Path(config["perspective_question_output"]).exists())
                self.assertFalse(Path(config["perspective_summary_output"]).exists())
                self.assertTrue(Path(config["one_agent_metrics_output"]).exists())
                self.assertTrue(Path(config["one_agent_model_subcategory_output"]).exists())
                self.assertTrue(Path(config["one_agent_summary_output"]).exists())
                self.assertTrue((Path(config["plots_dir"]) / "one_agent_accuracy_rate.svg").exists())
                self.assertTrue((Path(config["plots_dir"]) / "one_agent_instability_rate.svg").exists())
            else:
                self.assertTrue((Path(config["plots_dir"]) / "avg_corrected_after_wrong_rate.svg").exists())
                self.assertTrue((Path(config["plots_dir"]) / "avg_stood_firm_wrong_rate.svg").exists())
                self.assertTrue(Path(config["perspective_question_output"]).exists())
                self.assertTrue(Path(config["perspective_summary_output"]).exists())

            metrics_rows = self._read_csv(Path(config["metrics_output"])) if Path(config["metrics_output"]).exists() else []
            agg_rows = self._read_csv(Path(config["aggregate_output"])) if Path(config["aggregate_output"]).exists() else []
            return metrics_rows, agg_rows, config

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
