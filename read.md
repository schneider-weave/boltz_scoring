# Nanobody Scoring against P05231

Target and MSA match the validator setup in [metanova-labs/nova](https://github.com/metanova-labs/nova):
- Target: `P05231` (Interleukin-6, human)
- Clip interval: `[30, 212]` → 182-residue scoring sequence
- MSA: `data/msa_files/P05231.a3m` from [nova/data/msa_files](https://github.com/metanova-labs/nova/tree/main/data/msa_files)

Regenerate scoring inputs after every target change; previously generated YAMLs retain the old target.

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
| `scoring_results/intermediate_designs/aggregate_metrics_analyze.csv` | Full boltzgen metrics, one row per replicate |
| `scoring_results/validator_metrics.csv` | 10 validator metrics (median) + `_sd` + ranks + `rank_sum` |
| `scoring_results/validator_metrics_long.csv` | One row per metric per design |

The script runs `--steps design folding analysis`. boltzgen 0.2.0 builds `design_folding`
only for `protein-anything`/`protein-small_molecule`, so the validator's own
`nanobody-anything` run silently skips it even though nova's `boltzgen_config.yaml`
lists it in `execute_steps`; the ranked metrics all come from the `folding` step.
`filtering` is skipped, matching NOVA.

Overrides: `NUM_DESIGNS` (replicates, default 3), `SEED` (default 0),
`USE_KERNELS` (`auto`/`true`/`false`, default `auto` — set `false` if
cuequivariance kernels fail to load).

Manual boltzgen only (no export):

```bash
boltzgen run scoring_inputs/ \
    --output scoring_results/ \
    --protocol nanobody-anything \
    --skip_inverse_folding \
    --num_designs 3 \
    --seed 0 \
    --steps design folding analysis \
    --step_scale 2.0 \
    --noise_scale 0.88 \
    --cache /workspace/cache \
    --use_kernels auto
```

## 3a. Two-stage selection (recommended)

Score everything once to discard the clearly bad designs, pick the survivors by
hand, then score only those three times and submit the winner.

Both stages take a plain-text sequence file: one sequence per line, `#` lines
ignored. Use two different files so the screening pool survives the selection.

```bash
# Stage 1 — screen: one replicate per sequence
rm -rf scoring_inputs/
python generate_scoring_yamls.py --input all_sequences.txt --output_dir scoring_inputs/
NUM_DESIGNS=1 bash scripts/run_scoring.sh scoring_inputs/ screen_results/

# Read screen_results/validator_metrics.csv (sorted, lowest rank_sum first) and
# paste the sequences you want to keep into nanobodies.txt, one per line.
# The designed_sequence column holds them.

# Stage 2 — confirm: three replicates per sequence (NUM_DESIGNS=3 is the default)
rm -rf finalists/
python generate_scoring_yamls.py --input nanobodies.txt --output_dir finalists/
bash scripts/run_scoring.sh finalists/ final_results/
```

`generate_scoring_yamls.py` auto-detects the format, so `.fasta` and `.csv`
inputs work the same way if you ever need them.

Submit the top row of `final_results/validator_metrics.csv` (lowest `rank_sum`).

Keep 30–50% of the screened designs. Cutting a 50-design batch to the top 10%
retains only ~32% of the truly-best designs; keeping 40–50% retains 75–83%.

`rm -rf finalists/` matters: boltzgen scores **every** YAML in the input
directory, so leftovers from a previous round would be scored again.

Design ids are `nb<line>_h<md5-of-sequence>`. The `nb0000` part renumbers when
the input file changes, but the `h...` hash is stable — use it to match a
stage-2 row back to its stage-1 row.

Run stage 2 as **one batch on one device**: `rank_sum` is relative to whichever
designs share the CSV, so it cannot be compared across separate runs or devices.

## 3b. Why 3 replicates

The design and folding steps are diffusion samplers, so each run draws a different
structure. Measured single-run scatter is as large as the real spread between similar
designs (`design_iiptm` ±0.015, `plip_hbonds` ±3, `delta_sasa_refolded` ±46), which is
why one run cannot order near-identical sequences. Scoring 3× and ranking the median
raises the chance of picking the true best design from ~55% to ~69%, and puts it in the
local top 3 about 99% of the time.

Read `validator_metrics.csv` as: **lowest `rank_sum` wins**, but two designs differ
meaningfully only when a metric gap exceeds roughly 3× its `_sd` column. Treat the top
few as a tied group rather than a strict ordering.

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
