BASE_CONFIG ?= configs/analysis_config.yaml
ACTIVE_CONFIG_FILE ?= configs/.active_run_config
ACTIVE_CONFIG_PATH ?= $(shell test -f $(ACTIVE_CONFIG_FILE) && cat $(ACTIVE_CONFIG_FILE) || echo "")
CONFIG ?= $(shell test -n "$(ACTIVE_CONFIG_PATH)" && test -f "$(ACTIVE_CONFIG_PATH)" && echo "$(ACTIVE_CONFIG_PATH)" || echo $(BASE_CONFIG))
PYTHON ?= python3
MPLCONFIGDIR ?= .cache/matplotlib
SAMPLE_JSONL ?=
HF_REVISION ?=
SCHEMA_PROFILE ?=
RUN_LABEL ?=
RAW_ROOT ?= raw
HF_INCLUDE ?=

export MPLCONFIGDIR

.PHONY: all dirs preflight metadata normalize one_agent legacy perspective metrics aggregate plots report prepare_batch download_hf ingest_hf show_config test clean

all: dirs preflight metadata normalize one_agent legacy perspective metrics aggregate plots report

dirs:
	mkdir -p raw intermediate metrics plots reports configs/runs

show_config:
	@echo "Using config: $(CONFIG)"

preflight:
	$(PYTHON) scripts/preflight_validate.py --config $(CONFIG)

metadata:
	$(PYTHON) scripts/write_run_metadata.py --config $(CONFIG)

normalize:
	$(PYTHON) scripts/normalize_results.py --config $(CONFIG)

one_agent:
	$(PYTHON) scripts/compute_one_agent_metrics.py --config $(CONFIG)

legacy:
	$(PYTHON) scripts/analyze_legacy_compat.py --config $(CONFIG)

perspective:
	$(PYTHON) scripts/compute_perspective_metrics.py --config $(CONFIG)

metrics:
	$(PYTHON) scripts/compute_behavior_metrics.py --config $(CONFIG)

aggregate:
	$(PYTHON) scripts/aggregate_metrics.py --config $(CONFIG)

plots:
	$(PYTHON) scripts/plot_metrics.py --config $(CONFIG)

report:
	$(PYTHON) scripts/build_summary_report.py --config $(CONFIG)

prepare_batch:
	$(PYTHON) scripts/prepare_batch.py --config $(BASE_CONFIG) --sample-jsonl "$(SAMPLE_JSONL)" --hf-revision "$(HF_REVISION)" --schema-profile "$(SCHEMA_PROFILE)" --run-label "$(RUN_LABEL)" --raw-root "$(RAW_ROOT)" --run-config-dir configs/runs --active-config-file $(ACTIVE_CONFIG_FILE)

download_hf:
	$(PYTHON) scripts/download_hf_batch.py --config $(CONFIG) --include "$(HF_INCLUDE)"

ingest_hf:
	$(PYTHON) scripts/prepare_batch.py --config $(BASE_CONFIG) --sample-jsonl "$(SAMPLE_JSONL)" --hf-revision "$(HF_REVISION)" --schema-profile "$(SCHEMA_PROFILE)" --run-label "$(RUN_LABEL)" --raw-root "$(RAW_ROOT)" --run-config-dir configs/runs --active-config-file $(ACTIVE_CONFIG_FILE)
	$(PYTHON) scripts/download_hf_batch.py --config "$$(cat $(ACTIVE_CONFIG_FILE))" --include "$(HF_INCLUDE)"

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

clean:
	find intermediate -type f -name '*.jsonl' -delete
	find metrics -type f -name '*.csv' -delete
	find plots -type f -name '*.svg' -delete
	find reports -type f \\( -name '*.md' -o -name '*.json' \\) -delete
