#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Iluvatar CoreX CI environment.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH:-}"

: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    VLLM_PLUGINS \
    USE_FLAGGEMS \
    GEMS_VENDOR; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name}=${!name}" >> "${GITHUB_ENV}"
    fi
  done
fi

# vLLM / FlagGems / torch come from the (future) CI image.
# Install only the checked-out plugin source for this workflow run.
python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import flag_gems
import torch
import vllm
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems vendor: {getattr(flag_gems, 'vendor_name', 'auto-detected')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY
