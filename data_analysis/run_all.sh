#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/analysis_config.yaml}"
make -C data_analysis all CONFIG="$CONFIG_PATH"
