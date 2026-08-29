#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
jupyter nbconvert \
  --to notebook \
  --execute tesis_eda_v11.ipynb \
  --output tesis_eda_v11_executed.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
