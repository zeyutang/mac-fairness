from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


class PrepareBatchTest(unittest.TestCase):
    def test_prepare_batch_infers_label_and_creates_raw_dir(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_hf_one_agent.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg = tmp_dir / "analysis_config.yaml"
            raw_root = tmp_dir / "raw"
            cfg.write_text(
                "\n".join(
                    [
                        "run_label: default",
                        "schema_profile: auto",
                        "raw_glob: raw/{hf_dataset_revision}/{run_label}/*.jsonl",
                        "normalized_output: intermediate/{run_label}/normalized.jsonl",
                        "run_metadata_output: reports/{run_label}/run_metadata.json",
                        "metrics_output: metrics/{run_label}/behavior_metrics.csv",
                        "aggregate_output: metrics/{run_label}/behavior_metrics_by_model.csv",
                        "report_output: reports/{run_label}/summary.md",
                        "plots_dir: plots/{run_label}",
                        "hf_dataset_repo: https://huggingface.co/datasets/aims-foundation/mac-fairness",
                        "hf_dataset_revision: \"\"",
                        "hf_snapshot_date: \"\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "python3",
                    "scripts/prepare_batch.py",
                    "--config",
                    str(cfg),
                    "--sample-jsonl",
                    str(fixture),
                    "--hf-revision",
                    "a70d430",
                    "--date",
                    "2026-02-24",
                    "--raw-root",
                    str(raw_root),
                    "--in-place",
                ],
                cwd=repo_root / "data_analysis",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            text = cfg.read_text(encoding="utf-8")
            self.assertIn("run_label: 2026-02-24_bbq_age_1agent", text)
            self.assertIn("schema_profile: one_agent", text)
            self.assertIn("hf_dataset_revision: a70d430", text)
            self.assertTrue((raw_root / "a70d430" / "2026-02-24_bbq_age_1agent").exists())

    def test_prepare_batch_writes_immutable_run_config_and_active_pointer(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        fixture = Path(__file__).resolve().parent / "fixtures" / "sample_hf_one_agent.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_cfg = tmp_dir / "base.yaml"
            run_cfg_dir = tmp_dir / "runs"
            active_ptr = tmp_dir / ".active_run_config"
            base_cfg.write_text(
                "\n".join(
                    [
                        "run_label: default",
                        "schema_profile: auto",
                        "raw_glob: raw/{hf_dataset_revision}/{run_label}/**/*.jsonl",
                        "normalized_output: intermediate/{run_label}/normalized.jsonl",
                        "run_metadata_output: reports/{run_label}/run_metadata.json",
                        "metrics_output: metrics/{run_label}/behavior_metrics.csv",
                        "aggregate_output: metrics/{run_label}/behavior_metrics_by_model.csv",
                        "report_output: reports/{run_label}/summary.md",
                        "plots_dir: plots/{run_label}",
                        "hf_dataset_repo: https://huggingface.co/datasets/aims-foundation/mac-fairness",
                        "hf_dataset_revision: \"\"",
                        "hf_snapshot_date: \"\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "python3",
                    "scripts/prepare_batch.py",
                    "--config",
                    str(base_cfg),
                    "--sample-jsonl",
                    str(fixture),
                    "--hf-revision",
                    "a70d430",
                    "--date",
                    "2026-02-24",
                    "--run-config-dir",
                    str(run_cfg_dir),
                    "--active-config-file",
                    str(active_ptr),
                ],
                cwd=repo_root / "data_analysis",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            run_cfg = run_cfg_dir / "2026-02-24_bbq_age_1agent.yaml"
            self.assertTrue(run_cfg.exists())
            self.assertEqual(active_ptr.read_text(encoding="utf-8").strip(), str(run_cfg))


if __name__ == "__main__":
    unittest.main()
