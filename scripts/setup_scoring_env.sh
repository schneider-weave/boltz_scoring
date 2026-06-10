#!/usr/bin/env bash
# BoltzGen nanobody scoring environment for Vast.ai (CUDA 12.4, single GPU).
#
# Fixes the full failure chain seen on vastai/pytorch templates:
#   - libnvrtc-builtins (cu130 torch on cu124 driver)
#   - torchvision::nms (torch/torchvision version skew)
#   - ncclCommResume (torch 2.6 linked against newer NCCL than pip installed)
#   - numpy 2.4 vs boltzgen pin 2.0.2
#   - nvidia-cublas-cu12 12.4 vs cuequivariance >=12.5
#
# Usage (on Vast):
#   source /venv/main/bin/activate   # or: conda activate bg
#   cd /workspace/nova
#   bash scripts/setup_scoring_env.sh
#
# Prefer a FRESH env when /venv/main is badly polluted:
#   conda create -n bg python=3.12 -y && conda activate bg
#   bash scripts/setup_scoring_env.sh

set -euo pipefail

NOVA_DIR="${NOVA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"

# torch 2.5.1 avoids NCCL 2.29+ symbols (ncclCommResume) that break on mixed installs.
# boltzgen requires torch>=2.4.1 — 2.5.1 is the stable cu124 choice for RTX 3090.
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
NUMPY_VERSION="2.0.2"

echo "=============================================="
echo " BoltzGen scoring environment setup"
echo " Nova:  ${NOVA_DIR}"
echo " Python: $(python3 -c 'import sys; print(sys.executable)')"
echo "=============================================="

# System NCCL/CUDA on LD_LIBRARY_PATH often overrides pip nvidia-* wheels → undefined symbols.
unset LD_LIBRARY_PATH 2>/dev/null || true

echo ""
echo "--- Diagnostics (before fix) ---"
python3 - <<'DIAG' 2>/dev/null || true
import subprocess, sys
for pkg in ("torch", "torchvision", "numpy", "nvidia-nccl-cu12", "nvidia-cublas-cu12"):
    r = subprocess.run([sys.executable, "-m", "pip", "show", pkg], capture_output=True, text=True)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.startswith(("Name:", "Version:", "Location:")):
                print(line)
DIAG
if command -v conda >/dev/null 2>&1; then
  echo "conda torch packages:"
  conda list 2>/dev/null | grep -iE '^torch|^pytorch' || echo "  (none)"
fi

echo ""
echo "--- Step 1: Remove broken / conflicting torch CUDA stack ---"
TORCH_PACKAGES=(
  torch torchvision torchaudio torchmetrics triton
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12
  nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cusparselt-cu12
  nvidia-cuda-cupti-cu12
)
pip uninstall -y "${TORCH_PACKAGES[@]}" 2>/dev/null || true

# conda pytorch alongside pip torchvision is a common cause of ::nms errors
if command -v conda >/dev/null 2>&1; then
  conda remove -y pytorch torchvision torchaudio pytorch-cuda 2>/dev/null || true
fi

echo ""
echo "--- Step 2: Pin numpy (boltzgen==2.0.2, numba<2.2) BEFORE torch ---"
pip install "numpy==${NUMPY_VERSION}"

echo ""
echo "--- Step 3: Install matched torch + torchvision from ONE index ---"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${PYTORCH_INDEX}"

echo ""
echo "--- Step 4: CUDA libs cuequivariance / torch expect (newer than torch's defaults) ---"
pip install \
  "nvidia-cublas-cu12>=12.5.0" \
  "nvidia-nccl-cu12>=2.21.5"

echo ""
echo "--- Step 5: Re-pin numpy (torch install may have upgraded it) ---"
pip install "numpy==${NUMPY_VERSION}"

echo ""
echo "--- Step 6: pytorch-lightning dependency ---"
pip install "torchmetrics>=0.7.0"

echo ""
echo "--- Step 7: RDKit (conda is reliable; skip if already importable) ---"
if ! python3 -c "import rdkit" 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    conda install -c conda-forge rdkit -y
  else
    echo "WARNING: rdkit not found and conda unavailable. Install rdkit manually."
  fi
fi

echo ""
echo "--- Step 8: Install boltzgen without letting pip upgrade numpy/torch ---"
pip install -e "${NOVA_DIR}/boltzgen/" --no-deps
pip install \
  "numba==0.61.0" matplotlib hydride biotite pydssp logomaker \
  "gemmi>=0.6.5" scikit-learn hydra-core edit_distance pytorch-lightning \
  pandas pdbeccdutils einx einops mashumaro \
  "nvidia-ml-py>=12.535.133" \
  "cuequivariance_ops_cu12>=0.5.0" \
  "cuequivariance_ops_torch_cu12>=0.5.0" \
  "cuequivariance_torch>=0.5.0" \
  huggingface_hub biopython

echo ""
echo "--- Step 9: Final numpy pin ---"
pip install "numpy==${NUMPY_VERSION}"

echo ""
echo "--- Verification ---"
python3 <<'VERIFY'
import sys
import numpy
import torch
import torchvision
import torchmetrics
import pytorch_lightning
import cuequivariance_ops_cu12

assert numpy.__version__ == "2.0.2", f"numpy must be 2.0.2, got {numpy.__version__}"
print("python:     ", sys.executable)
print("numpy:      ", numpy.__version__)
print("torch:      ", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchmetrics:", torchmetrics.__version__)
print("cuda:       ", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — check nvidia-smi and driver")
x = torch.randn(4, 3, 3, device="cuda")
print("cuda det:   ", torch.det(x).shape)
import boltzgen
print("boltzgen:   ", boltzgen.__file__)
print("")
print("ALL OK — environment ready for scoring.")
VERIFY

echo ""
echo "Run scoring:"
echo "  export OMP_NUM_THREADS=1"
echo "  cd ${NOVA_DIR}"
echo "  boltzgen run scoring_inputs_one/ --output scoring_results_one_fixed/ \\"
echo "    --protocol nanobody-anything --skip_inverse_folding --num_designs 1 \\"
echo "    --steps design folding analysis --step_scale 2.0 --noise_scale 0.88 \\"
echo "    --cache /workspace/cache"
