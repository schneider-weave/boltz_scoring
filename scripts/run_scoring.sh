#!/usr/bin/env bash
# Full nanobody scoring pipeline: boltzgen + validator metrics export.
#
# Scoring is stochastic, so every sequence is scored NUM_DESIGNS times and the
# export collapses the replicates to a median per sequence. Rank on that median;
# a single replicate is too noisy to order similar designs.
#
# Usage:
#   bash scripts/run_scoring.sh
#   bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
#   CACHE=/workspace/cache bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
#   NUM_DESIGNS=5 bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
#
# Environment overrides:
#   NUM_DESIGNS   replicates per sequence (default 3; 1 reproduces the old behaviour)
#   SEED          fixed RNG seed, so a rerun of this script reproduces itself (default 0)
#   USE_KERNELS   auto|true|false (default auto, which is what the validator uses).
#                 Set to false if cuequivariance kernels fail to load on this box.
#
# Outputs:
#   scoring_results/intermediate_designs/aggregate_metrics_analyze.csv  (full boltzgen, one row per replicate)
#   scoring_results/validator_metrics.csv                               (10 ranked metrics + _sd + ranks)
#   scoring_results/validator_metrics_long.csv                          (long format)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="${1:-scoring_inputs/}"
OUTPUT="${2:-scoring_results/}"
CACHE="${CACHE:-/workspace/cache}"
NUM_DESIGNS="${NUM_DESIGNS:-3}"
SEED="${SEED:-0}"
USE_KERNELS="${USE_KERNELS:-auto}"

# The validator pins a single GPU via CUDA_VISIBLE_DEVICES; matching that keeps
# batch composition (and therefore the scores) independent of local GPU count.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ANALYZE_CSV="${OUTPUT}/intermediate_designs/aggregate_metrics_analyze.csv"
VALIDATOR_CSV="${OUTPUT}/validator_metrics.csv"
VALIDATOR_LONG_CSV="${OUTPUT}/validator_metrics_long.csv"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "==> Scoring input:  ${INPUT}"
echo "==> Scoring output: ${OUTPUT}"
echo "==> Model cache:    ${CACHE}"
echo "==> Replicates:     ${NUM_DESIGNS} per sequence (seed ${SEED})"

# Steps match what the validator actually executes: boltzgen 0.2.0 builds
# design_folding only for protein-anything/protein-small_molecule, so the
# nanobody-anything run silently skips it despite nova's boltzgen_config.yaml
# listing it in execute_steps. The ranked metrics come from the folding step.
# trainer.deterministic=false is required: passing a seed makes predict.py default
# the trainer to deterministic=True, which calls torch.use_deterministic_algorithms(True)
# and raises RuntimeError on CUDA matmuls and index_add_/scatter_add_. The seed still
# fixes the RNG stream; only nondeterministic GPU reductions are left free.
boltzgen run "${INPUT}" \
  --output "${OUTPUT}" \
  --protocol nanobody-anything \
  --skip_inverse_folding \
  --num_designs "${NUM_DESIGNS}" \
  --seed "${SEED}" \
  --steps design folding analysis \
  --config design trainer.deterministic=false \
  --config folding trainer.deterministic=false \
  --step_scale 2.0 \
  --noise_scale 0.88 \
  --cache "${CACHE}" \
  --use_kernels "${USE_KERNELS}" \
  "${@:3}"

if [[ ! -f "${ANALYZE_CSV}" ]]; then
  echo "ERROR: Expected analyze CSV not found: ${ANALYZE_CSV}" >&2
  exit 1
fi

echo "==> Exporting validator ranking metrics..."
python3 "${REPO_ROOT}/scripts/export_validator_metrics.py" \
  "${ANALYZE_CSV}" \
  -o "${VALIDATOR_CSV}" \
  --long-output "${VALIDATOR_LONG_CSV}" \
  --include-sequence

echo "==> Done."
echo "    Full metrics:     ${ANALYZE_CSV}"
echo "    Validator scores: ${VALIDATOR_CSV}"
echo "    Validator long:   ${VALIDATOR_LONG_CSV}"
