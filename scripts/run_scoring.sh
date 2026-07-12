#!/usr/bin/env bash
# Full nanobody scoring pipeline: boltzgen (validator parity) + validator metrics export.
#
# Usage:
#   bash scripts/run_scoring.sh
#   bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
#   CACHE=/workspace/cache bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
#
# Outputs:
#   scoring_results/intermediate_designs/aggregate_metrics_analyze.csv  (full boltzgen)
#   scoring_results/validator_metrics.csv                               (10 ranked metrics + ranks)
#   scoring_results/validator_metrics_long.csv                          (long format)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="${1:-scoring_inputs/}"
OUTPUT="${2:-scoring_results/}"
CACHE="${CACHE:-/workspace/cache}"

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

boltzgen run "${INPUT}" \
  --output "${OUTPUT}" \
  --protocol nanobody-anything \
  --skip_inverse_folding \
  --validator-parity \
  --num_designs 1 \
  --step_scale 2.0 \
  --noise_scale 0.88 \
  --cache "${CACHE}" \
  --use_kernels false \
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
