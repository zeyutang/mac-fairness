from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


class DownloadHFBatchTest(unittest.TestCase):
    def test_download_command_dry_run_uses_resolved_raw_dir(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg = tmp_dir / "analysis_config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "run_label: 2026-02-24_bbq_age_1agent",
                        "schema_profile: one_agent",
                        "raw_glob: raw/{hf_dataset_revision}/{run_label}/*.jsonl",
                        "normalized_output: intermediate/{run_label}/normalized.jsonl",
                        "run_metadata_output: reports/{run_label}/run_metadata.json",
                        "metrics_output: metrics/{run_label}/behavior_metrics.csv",
                        "aggregate_output: metrics/{run_label}/behavior_metrics_by_model.csv",
                        "report_output: reports/{run_label}/summary.md",
                        "plots_dir: plots/{run_label}",
                        "hf_dataset_repo: https://huggingface.co/datasets/aims-foundation/mac-fairness",
                        "hf_dataset_revision: a70d430",
                        "hf_snapshot_date: 2026-02-24",
                        "hf_include_glob: *.jsonl",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python3",
                    "scripts/download_hf_batch.py",
                    "--config",
                    str(cfg),
                    "--dry-run",
                ],
                cwd=repo_root / "data_analysis",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertIn("download_hf_batch: local_dir=raw/a70d430/2026-02-24_bbq_age_1agent", proc.stdout)
            self.assertTrue(
                ("huggingface-cli download aims-foundation/mac-fairness" in proc.stdout)
                or ("hf download aims-foundation/mac-fairness" in proc.stdout)
            )


if __name__ == "__main__":
    unittest.main()
