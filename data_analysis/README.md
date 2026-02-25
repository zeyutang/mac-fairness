# Data Analysis Pipeline

This folder is a shared, reproducible pipeline for post-experiment analysis.

## Collaboration Workflow

1. Branch from `analysis/main` (`analysis/<name>-<task>`).
2. Add or modify scripts in `data_analysis/scripts/`.
3. Update metric definitions in `metrics_registry.md` if metric behavior changes.
4. Run `make -C data_analysis all` before opening a PR.
5. Open PR into `analysis/main` and require one reviewer.

## Directory Layout

- `raw/`: input JSONL files copied from experiment outputs or exported from Hugging Face.
- `intermediate/`: normalized canonical records.
- `metrics/`: metric CSV outputs.
- `plots/`: generated SVG charts.
- `reports/`: run metadata and markdown summary.
- `configs/`: pipeline config files.
- `scripts/`: pipeline scripts.
- `tests/`: fixture-based regression tests.

## Batch Isolation

Use one raw folder per pull batch:

- `raw/<hf_revision>/<run_label>/**/*.jsonl`

Set `run_label`, `hf_dataset_revision`, and `schema_profile` in config for that batch. Outputs are auto-separated by `run_label`.
Track pulls in `raw_manifest.csv` (one row per pull batch).
Prepared run configs are immutable and written to `configs/runs/<run_label>.yaml`.
The active config pointer is `configs/.active_run_config`.

## Data Contract

`normalize_results.py` emits canonical JSONL records with these required fields:

- `run_id` (str)
- `model` (str)
- `question_id` (str)
- `answer_before` (str)
- `answer_after` (str)
- `correct_answer` (str)
- `changed_answer` (bool)
- `initial_is_correct` (bool)
- `final_is_correct` (bool)
- `has_initial_answer` (bool)

If raw rows are missing required values, they are skipped and counted in script logs.

For MAC transcript exports, the normalizer supports `vanilla_agent_*` and `identity_agent_*` fields and emits one normalized row per agent.
For one-agent exports (`final_answer`, `is_correct`), rows are normalized with `has_initial_answer=false`, and change-based metrics are excluded from denominators.

Perspective metrics (`volatility`, `stubbornness`, `agreeableness`, and identity-minus-vanilla deltas) are computed directly from multi-agent raw rows with `identity_agent_r*` and `vanilla_agent_r*` fields.
For one-agent runs, the pipeline computes one-agent metrics (`accuracy_rate`, BBQ bias metrics such as `ambig_bias_score`/`disambig_bias_score`, and cross-condition `instability_rate`) and skips multi-agent-only metrics.

## Commands

## New User: Run Analysis End-to-End

1. Open the repo root:
```bash
cd mac-fairness
```

2. Authenticate once with Hugging Face (required for private dataset pulls):
```bash
hf auth login
```

3. Prepare a new batch config and run label (this writes `configs/runs/<run_label>.yaml` and updates `configs/.active_run_config`):
```bash
make -C data_analysis prepare_batch HF_REVISION=main SCHEMA_PROFILE=one_agent RUN_LABEL=0260224t175803_529z_qwen3-30b_1agent_as-ai_v2025-12-10
```

4. Download raw JSONL files into the batch raw folder:
```bash
make -C data_analysis download_hf HF_INCLUDE="sampled-set-with-rep-exps/bbq-sample/*.jsonl"
```

5. Run the full pipeline:
```bash
make -C data_analysis all
```

6. Read outputs for this run label:
- `data_analysis/metrics/<run_label>/`
- `data_analysis/plots/<run_label>/`
- `data_analysis/reports/<run_label>/summary.md`

7. (Optional) Run tests before pushing:
```bash
make -C data_analysis test
```

## Add a New Metric (Step-by-Step)

1. Decide the metric scope:
- `one_agent` metric: add to `scripts/compute_one_agent_metrics.py`
- multi-agent normalized metric: add to `scripts/compute_behavior_metrics.py`
- identity-vs-vanilla perspective metric: add to `scripts/compute_perspective_metrics.py`

2. Implement the metric calculation in the correct script and add the metric as a new CSV column in that script's output writer.

3. If the metric needs rollups, update downstream script(s):
- model rollups: `scripts/aggregate_metrics.py` or one-agent summary logic
- legacy one-agent comparisons: `scripts/analyze_legacy_compat.py`
- reporting headline: `scripts/build_summary_report.py` (optional)
- plotting: `scripts/plot_metrics.py` (optional)

4. Register the metric definition in `data_analysis/metrics_registry.md`:
- metric name
- definition/formula
- owner
- script
- output column

5. Add or update tests under `data_analysis/tests/`:
- assert output file exists
- assert new column exists
- assert at least one expected value from fixture data

6. Run the full pipeline locally on a fixture or batch:
```bash
make -C data_analysis all
```

7. Run tests:
```bash
make -C data_analysis test
```

8. Update docs if behavior changed:
- `data_analysis/README.md`
- `data_analysis/metrics_registry.md`

Prepare a new batch config/folder (auto run label):

```bash
make -C data_analysis prepare_batch HF_REVISION=a70d430 SAMPLE_JSONL=tests/fixtures/sample_hf_one_agent.jsonl
```

Optional overrides:
- `RUN_LABEL=...` to force a manual label
- `SCHEMA_PROFILE=one_agent|multi_agent|generic`
- without `SAMPLE_JSONL`, auto label defaults to `<date>_hfpull_<revision7>`

Prepare + download from Hugging Face in one command:

```bash
make -C data_analysis ingest_hf HF_REVISION=a70d430
```

This command:
- writes immutable run config to `configs/runs/<run_label>.yaml`
- updates active config pointer at `configs/.active_run_config`
- creates the raw batch folder
- downloads `*.jsonl` from `hf_dataset_repo` at `hf_dataset_revision` into that folder

Prerequisite: run `hf auth login` (or `huggingface-cli login`) once on your machine.

Run full pipeline:

```bash
make -C data_analysis all
```

Pipeline includes:
- preflight validation (fails if raw files are missing or schema/profile mismatch is severe)
- metadata manifest with raw-file checksums
- schema-aware metric stages (one-agent vs multi-agent)
- legacy-compatible one-agent condition analysis outputs

Perspective outputs:
- `metrics/<run_label>/perspective_question.csv`
- `metrics/<run_label>/perspective_summary.csv`

One-agent outputs:
- `metrics/<run_label>/one_agent_metrics.csv`
- `metrics/<run_label>/one_agent_model_subcategory_scores.csv` (rolled up by `model + benchmark_subcategory`)
- `metrics/<run_label>/one_agent_summary_by_model.csv`
- `metrics/<run_label>/legacy_analysis/*.csv` (legacy-compatible condition summaries/win-rates/pairwise deltas)

One-agent plots (under `plots/<run_label>/`):
- `one_agent_accuracy_rate.svg`
- `one_agent_instability_rate.svg`
- `one_agent_accuracy_vs_instability_scatter.svg`
- `one_agent_bias_tradeoff_scatter.svg`

Run only tests:

```bash
make -C data_analysis test
```

Use a custom config:

```bash
make -C data_analysis all CONFIG=configs/analysis_config.yaml
```

## Troubleshooting

1. `FileNotFoundError ... /var/folders/.../tmp.../analysis_config.yaml`
- Cause: `configs/.active_run_config` points to a temporary test config that no longer exists.
- Fix:
```bash
echo "configs/runs/<your_run_label>.yaml" > data_analysis/configs/.active_run_config
make -C data_analysis show_config
```
- Alternative: bypass active pointer and run with explicit config:
```bash
make -C data_analysis all CONFIG=configs/runs/<your_run_label>.yaml
```

2. `matplotlib not available` or packages not found during `make`
- Cause: wrong Python interpreter/environment.
- Fix (recommended):
```bash
make -C data_analysis all PYTHON=.venv/bin/python3
```
- Or activate venv first:
```bash
source .venv/bin/activate
make -C data_analysis all
```

3. HF download command succeeds but analysis outputs are empty
- Cause: no JSONL files were actually downloaded into the active raw folder (often wrong include glob/path).
- Fix:
```bash
make -C data_analysis download_hf HF_INCLUDE="sampled-set-with-rep-exps/bbq-sample/*.jsonl"
find data_analysis/raw -type f -name "*.jsonl" | head
```

4. Metrics/plots look empty or many stages are skipped
- Cause: schema/profile mismatch (`one_agent` vs `multi_agent`).
- Fix:
```bash
make -C data_analysis prepare_batch SCHEMA_PROFILE=one_agent HF_REVISION=main RUN_LABEL=<run_label>
make -C data_analysis show_config
```
Confirm `schema_profile` in the active config matches your dataset format.

5. Multiple output folders that look similar
- Cause: different `run_label` values create separate folders by design.
- Fix: pick one stable run label convention and rerun with that label.

6. `Matplotlib is building the font cache; this may take a moment`
- This is normal on first run in a new environment/cache.
