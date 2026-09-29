#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Iluvatar CoreX BI-V150 availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Iluvatar CoreX availability ==="

if command -v ixsmi >/dev/null 2>&1; then
  ixsmi || true
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::ixsmi not found; falling back to nvidia-smi (CoreX compat)."
  nvidia-smi || true
else
  echo "::warning::Neither ixsmi nor nvidia-smi found; probing via torch."
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Iluvatar/CUDA device is not available via torch.cuda")

count = torch.cuda.device_count()
print(f"Accelerator count: {count}")
if count < 1:
    raise RuntimeError("At least 1 accelerator is required")

name = torch.cuda.get_device_name(0)
print(f"Device 0: {name}")
tensor = torch.ones((32, 32), device="cuda:0")
torch.cuda.synchronize()
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
