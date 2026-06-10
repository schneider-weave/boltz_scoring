# Nanobody Scoring against P20809

Target and MSA match the validator setup in [metanova-labs/nova](https://github.com/metanova-labs/nova):
- Target: `P20809` (Interleukin-11, human)
- Clip interval: `[36, 199]` → 163-residue scoring sequence
- MSA: `data/msa_files/P20809.a3m` from [nova/data/msa_files](https://github.com/metanova-labs/nova/tree/main/data/msa_files)

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

## 3. Run boltzgen scoring (validator-like)

```bash
boltzgen run scoring_inputs/ \
    --output scoring_results/ \
    --protocol nanobody-anything \
    --skip_inverse_folding \
    --num_designs 1 \
    --steps design folding analysis \
    --step_scale 2.0 \
    --noise_scale 0.88
```

- `--skip_inverse_folding` — sequences are already fixed, skip inverse folding
- `--num_designs 1` — one structure per input (scoring mode)
- `--step_scale` / `--noise_scale` — match validator `boltzgen_config.yaml`
- Models (~6 GB) download automatically to `~/.cache` on first run

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
