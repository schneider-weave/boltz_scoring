# Nanobody Scoring against P01584

Target and MSA match the validator setup in [metanova-labs/nova](https://github.com/metanova-labs/nova):
- Target: `P01584` (Interleukin-1 beta, human)
- Clip interval: `[116, 269]` → 153-residue scoring sequence
- MSA: `data/msa_files/P01584.a3m` from [nova/data/msa_files](https://github.com/metanova-labs/nova/tree/main/data/msa_files)

Do **not** use `scoring_inputs_fixed/` — those YAMLs are outdated (P05231).

## 1. Environment Setup

On **Vast.ai**, do not use the template's default PyTorch 2.10+cu130 — it breaks NVRTC,
and repeated `pip install torch` cycles create NCCL/numpy/torchvision conflicts.

Use a **fresh env** and the pinned installer:

```bash
# Recommended: clean conda env (not polluted /venv/main)
conda create -n bg python=3.12 -y
conda activate bg

cd /workspace/nova   # or your clone path
bash scripts/setup_scoring_env.sh
```

The script installs **torch 2.5.1 + torchvision 0.20.1 (cu124)**, **numpy 2.0.2**,
and the CUDA libs cuequivariance needs. It verifies `import torch` and a CUDA `det()` before finishing.

If you must reuse `/venv/main`, still run `bash scripts/setup_scoring_env.sh` — it removes
the broken torch/NCCL stack first.

Manual install (not recommended on Vast):

```bash
conda create -n bg python=3.12 -y
conda activate bg
conda install -c conda-forge rdkit -y
cd boltzgen && pip install -e . && cd ..
```

## 2. Generate YAML input files from filter_passed.fasta

```bash
python generate_scoring_yamls.py \
    --input filter_passed.fasta \
    --output_dir scoring_inputs/
```

This reads every sequence from `filter_passed.fasta` and writes one YAML file
per nanobody into `scoring_inputs/`. The FASTA header (e.g. `design_spec_0673|rank=4`)
becomes the filename: `design_spec_0673_rank_4.yaml`.

## 3. Run full scoring pipeline (validator parity + export)

One command runs boltzgen and writes `validator_metrics.csv` automatically:

```bash
rm -rf scoring_results/

CACHE=/workspace/cache bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
```

Or with explicit thread limits (recommended on shared GPUs):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CACHE=/workspace/cache \
bash scripts/run_scoring.sh scoring_inputs/ scoring_results/
```

**Outputs:**
| File | Contents |
|---|---|
| `scoring_results/intermediate_designs/aggregate_metrics_analyze.csv` | Full boltzgen metrics |
| `scoring_results/validator_metrics.csv` | 10 validator metrics + ranks + `rank_sum` |
| `scoring_results/validator_metrics_long.csv` | One row per metric per design |

The script uses `--validator-parity` (design → folding → design_folding → analysis, no fixed seed). Do **not** pass `--steps design folding analysis` — that skips `design_folding`.

Manual boltzgen only (no export):

```bash
boltzgen run scoring_inputs/ \
    --output scoring_results/ \
    --protocol nanobody-anything \
    --skip_inverse_folding \
    --validator-parity \
    --num_designs 1 \
    --step_scale 2.0 \
    --noise_scale 0.88 \
    --cache /workspace/cache \
    --use_kernels false
```

## 4. Results

Scores are in:
```
scoring_results/intermediate_designs/aggregate_metrics_analyze.csv
```

Key metrics per nanobody:
| Metric | Meaning |
|---|---|
| `design_ptm` | Intra-design TM score (validator confidence metric) |
| `design_to_target_iptm` | Design–target interface TM score |
| `min_design_to_target_pae` | Min PAE at interface (lower = better) |
| `delta_sasa_refolded` | Buried surface area (higher = better binding) |
| `plip_hbonds_refolded` | H-bonds at interface |
| `liability_score` | Developability score (lower = better) |

## Input file formats supported by generate_scoring_yamls.py

| Format | Example |
|---|---|
| FASTA | `filter_passed.fasta` — `>header` then sequence |
| CSV | `nanobodies.csv` — requires `sequence` column, optional `id` column |
| Plain text | `nanobodies.txt` — one sequence per line, `#` lines ignored |
