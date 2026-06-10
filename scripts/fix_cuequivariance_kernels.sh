#!/usr/bin/env bash
# Fix cuEquivariance GPU kernels (triangle_multiplicative_update) on Vast.
# Run when scoring works with --use_kernels false but fails with auto/true.
#
# Usage:
#   source /venv/main/bin/activate   # or: conda activate bg
#   cd /workspace/nova
#   bash scripts/fix_cuequivariance_kernels.sh

set -euo pipefail

PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
NUMPY_VERSION="2.0.2"
NOVA_DIR="${NOVA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

CONSTRAINTS="$(mktemp)"
trap 'rm -f "${CONSTRAINTS}"' EXIT
cat > "${CONSTRAINTS}" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
numpy==${NUMPY_VERSION}
EOF

echo "=== Fix cuEquivariance kernels ==="

# Step 1: Reinstall cuequivariance WITH all deps (core package was missing before)
pip uninstall -y \
  cuequivariance cuequivariance_torch \
  cuequivariance_ops_cu12 cuequivariance_ops_torch_cu12 2>/dev/null || true

pip install \
  "cuequivariance>=0.5.0" \
  "cuequivariance_torch>=0.5.0" \
  "cuequivariance_ops_cu12>=0.5.0" \
  "cuequivariance_ops_torch_cu12>=0.5.0" \
  -c "${CONSTRAINTS}"

# Step 2: Re-pin torch (cuequivariance can pull torch 2.12 from PyPI without this)
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${PYTORCH_INDEX}" \
  --force-reinstall --no-deps

# Triton is required by cuequivariance_ops_torch but omitted when torch uses --no-deps
pip install "triton>=3.1.0,<3.3"

# Step 3: CUDA libs — libcue_ops.so needs cublas >= 12.5 and nvrtc
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

# Step 4: Write scoring_env.sh with LD_LIBRARY_PATH for native libs
ENV_HELPER="${NOVA_DIR}/scripts/scoring_env.sh"
python3 - <<PY > "${ENV_HELPER}"
import glob, os, site

def cuda_lib_paths():
    paths = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if not sp or not os.path.isdir(sp):
            continue
        paths.extend(glob.glob(os.path.join(sp, "cuequivariance_ops*", "lib")))
        paths.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    # torch 2.5.1+cu124 — skip cu13 wheels that conflict with cu12 libs
    paths = [p for p in dict.fromkeys(paths) if os.path.isdir(p) and "/nvidia/cu13/" not in p]
    return paths

paths = cuda_lib_paths()
print("# Source before boltzgen: source scripts/scoring_env.sh")
print("export OMP_NUM_THREADS=1")
if paths:
    print("export LD_LIBRARY_PATH=" + ":".join(paths) + ':${LD_LIBRARY_PATH:-}')
else:
    print("# WARNING: no CUDA/cuequivariance lib dirs found under site-packages")
PY
chmod +x "${ENV_HELPER}"
echo "Wrote ${ENV_HELPER}"
echo "Library paths:"
grep LD_LIBRARY_PATH "${ENV_HELPER}" || true

# Step 5: Verify kernel import
# shellcheck disable=SC1090
source "${ENV_HELPER}"

python3 - <<'VERIFY'
import sys
import torch

print("torch:", torch.__version__)
assert torch.__version__.startswith("2.5.1"), f"wrong torch: {torch.__version__}"

import triton
print("triton:", triton.__version__)

# Use boltzgen's own module so weight shapes match the kernel API
from boltzgen.model.layers.triangular import TriangleMultiplicationOutgoing

m = TriangleMultiplicationOutgoing(dim=128).cuda().eval()
x = torch.randn(1, 32, 32, 128, device="cuda")
mask = torch.ones(1, 32, 32, dtype=torch.bool, device="cuda")

with torch.no_grad():
    out_kernel = m(x, mask, use_kernels=True)
    out_ref = m(x, mask, use_kernels=False)

print("kernel output shape:", out_kernel.shape)
print("max |kernel - ref|:", (out_kernel - out_ref).abs().max().item())
print("kernel forward pass: OK")
VERIFY

echo ""
echo "=== Kernels fixed. Run scoring with: ==="
echo "  source ${NOVA_DIR}/scripts/scoring_env.sh"
echo "  boltzgen run scoring_inputs_one/ --output scoring_results/ \\"
echo "    --protocol nanobody-anything --skip_inverse_folding --num_designs 1 \\"
echo "    --steps design folding analysis --step_scale 2.0 --noise_scale 0.88 \\"
echo "    --cache /workspace/cache --use_kernels auto"
