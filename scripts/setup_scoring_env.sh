#!/usr/bin/env bash
# BoltzGen nanobody scoring environment for Vast.ai (CUDA 12.4, single GPU).
#
# Usage:
#   conda create -n bg python=3.12 -y && conda activate bg
#   cd /workspace/nova && bash scripts/setup_scoring_env.sh

set -euo pipefail

NOVA_DIR="${NOVA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"

TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
NUMPY_VERSION="2.0.2"

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
numpy==${NUMPY_VERSION}
EOF

echo "=============================================="
echo " BoltzGen scoring environment setup"
echo " Nova:  ${NOVA_DIR}"
echo " Python: $(python3 -c 'import sys; print(sys.executable)')"
echo "=============================================="

unset LD_LIBRARY_PATH 2>/dev/null || true

install_torch_stack() {
  echo ""
  echo ">>> Re-pinning torch==${TORCH_VERSION} + torchvision==${TORCHVISION_VERSION} (cu124)"
  pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${PYTORCH_INDEX}" \
    --force-reinstall \
    --no-deps
  pip install \
    "nvidia-cublas-cu12>=12.5.0" \
    "nvidia-cuda-runtime-cu12" \
    "nvidia-cuda-nvrtc-cu12" \
    "nvidia-cudnn-cu12" \
    "nvidia-cufft-cu12" \
    "nvidia-curand-cu12" \
    "nvidia-cusolver-cu12" \
    "nvidia-cusparse-cu12" \
    "nvidia-nccl-cu12>=2.21.5" \
    "nvidia-nvjitlink-cu12" \
    "nvidia-nvtx-cu12" \
    --no-deps
  pip install "numpy==${NUMPY_VERSION}"
  # Triton required by cuequivariance kernels; not bundled when torch installed with --no-deps
  pip install "triton>=3.1.0,<3.3"
}

assert_torch_version() {
  python3 - <<'PY'
import sys, torch
v = torch.__version__
if not v.startswith("2.5.1"):
    sys.exit(f"torch version is {v}, expected 2.5.1+cu124 — a dependency upgraded torch")
print(f"torch OK: {v}")
PY
}

install_cuequivariance() {
  echo ""
  echo ">>> Installing cuEquivariance (with deps, torch pinned via constraints file)"
  pip uninstall -y \
    cuequivariance cuequivariance_torch \
    cuequivariance_ops_cu12 cuequivariance_ops_torch_cu12 2>/dev/null || true

  # MUST include cuequivariance core; --no-deps on ops packages alone breaks triangle kernels.
  pip install \
    "cuequivariance>=0.5.0" \
    "cuequivariance_torch>=0.5.0" \
    "cuequivariance_ops_cu12>=0.5.0" \
    "cuequivariance_ops_torch_cu12>=0.5.0" \
    -c "${CONSTRAINTS_FILE}"

  install_torch_stack
  assert_torch_version

  # libcublas >= 12.5 required by libcue_ops.so (cublasGemmGroupedBatchedEx)
  pip install "nvidia-cublas-cu12>=12.5.0" --no-deps
}

echo ""
echo "--- Step 1: Remove ALL torch / CUDA pip packages ---"
TORCH_PACKAGES=(
  torch torchvision torchaudio torchmetrics triton
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12
  nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cusparselt-cu12
  nvidia-cuda-cupti-cu12
)
pip uninstall -y "${TORCH_PACKAGES[@]}" 2>/dev/null || true

if command -v conda >/dev/null 2>&1; then
  conda remove -y pytorch torchvision torchaudio pytorch-cuda 2>/dev/null || true
fi

echo ""
echo "--- Step 2: numpy pin ---"
pip install "numpy==${NUMPY_VERSION}"

echo ""
echo "--- Step 3: torch stack ---"
install_torch_stack
assert_torch_version

echo ""
echo "--- Step 4: torchmetrics ---"
pip install "torchmetrics>=0.7.0" -c "${CONSTRAINTS_FILE}"
install_torch_stack
assert_torch_version

echo ""
echo "--- Step 5: RDKit ---"
if ! python3 -c "import rdkit" 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    conda install -c conda-forge rdkit -y
  else
    echo "WARNING: install rdkit manually (conda install -c conda-forge rdkit)"
  fi
fi

echo ""
echo "--- Step 6: cuEquivariance ---"
install_cuequivariance

echo ""
echo "--- Step 7: boltzgen + remaining deps ---"
pip install -e "${NOVA_DIR}/boltzgen/" -c "${CONSTRAINTS_FILE}"
install_torch_stack
assert_torch_version

echo ""
echo "--- Step 8: Write env helper for libcue_ops.so ---"
ENV_HELPER="${NOVA_DIR}/scripts/scoring_env.sh"
python3 - <<PY > "${ENV_HELPER}"
import glob, os, site
paths = []
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    if not sp or not os.path.isdir(sp):
        continue
    paths.extend(glob.glob(os.path.join(sp, "cuequivariance_ops*", "lib")))
    paths.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
paths = [p for p in dict.fromkeys(paths) if os.path.isdir(p) and "/nvidia/cu13/" not in p]
print("# Source before boltzgen: source scripts/scoring_env.sh")
print("export OMP_NUM_THREADS=1")
if paths:
    print("export LD_LIBRARY_PATH=" + ":".join(paths) + ':${LD_LIBRARY_PATH:-}')
else:
    print("# WARNING: cuequivariance / nvidia lib paths not found")
PY
chmod +x "${ENV_HELPER}"
echo "Wrote ${ENV_HELPER}"

echo ""
echo "--- Verification ---"
# shellcheck disable=SC1090
source "${ENV_HELPER}"
python3 <<'VERIFY'
import glob, os, site, sys
import numpy
import torch
import torchvision
import torchmetrics
import pytorch_lightning

if not torch.__version__.startswith("2.5.1"):
    raise SystemExit(f"FATAL: torch {torch.__version__} — expected 2.5.1+cu124")
if not numpy.__version__.startswith("2.0.2"):
    raise SystemExit(f"FATAL: numpy {numpy.__version__} — expected 2.0.2")

print("python:      ", sys.executable)
print("numpy:       ", numpy.__version__)
print("torch:       ", torch.__version__)
print("torchvision: ", torchvision.__version__)
print("cuda:        ", torch.cuda.is_available())
x = torch.randn(4, 3, 3, device="cuda")
print("cuda det:    ", torch.det(x).shape)

# Test cuequivariance kernels (optional — scoring works without via --use_kernels false)
try:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
    print("cuequivariance: triangle_multiplicative_update OK")
    KERNELS_OK = True
except ImportError as e:
    print("cuequivariance: FAILED —", e)
    print("  → use --use_kernels false when running boltzgen (slower but correct)")
    KERNELS_OK = False

import boltzgen
print("boltzgen:    ", boltzgen.__file__)
print("")
if KERNELS_OK:
    print("ALL OK — environment ready (kernels enabled).")
else:
    print("PARTIAL OK — run with:  source scripts/scoring_env.sh  and  --use_kernels false")
VERIFY

echo ""
echo "Run scoring:"
echo "  source ${NOVA_DIR}/scripts/scoring_env.sh"
echo "  cd ${NOVA_DIR}"
echo "  boltzgen run scoring_inputs_one/ --output scoring_results_one_fixed/ \\"
echo "    --protocol nanobody-anything --skip_inverse_folding --num_designs 1 \\"
echo "    --steps design folding analysis --step_scale 2.0 --noise_scale 0.88 \\"
echo "    --cache /workspace/cache --use_kernels false"
