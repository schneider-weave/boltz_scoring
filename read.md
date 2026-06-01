# Nanobody Scoring against P05231 (IL-6)

## 1. Environment Setup

```bash
conda create -n bg python=3.12
conda activate bg

cd boltzgen
pip install -e .
cd ..
```

## 2. Generate YAML input files from filter_passed.fasta

```bash
python generate_scoring_yamls.py \
    --input ../filter_passed.fasta \
    --output_dir scoring_inputs/
```

This reads every sequence from `filter_passed.fasta` and writes one YAML file
per nanobody into `scoring_inputs/`. The FASTA header (e.g. `design_spec_0673|rank=4`)
becomes the filename: `design_spec_0673_rank_4.yaml`.

## 3. Run boltzgen scoring

```bash
boltzgen run scoring_inputs/ \
    --output scoring_results/ \
    --protocol nanobody-anything \
    --skip_inverse_folding \
    --num_designs 1
```

- `--skip_inverse_folding` — sequences are already fixed, skip straight to folding + analysis
- `--num_designs 1` — one fold per input (scoring mode, not design mode)
- Models (~6 GB) download automatically to `~/.cache` on first run

## 4. Results

Scores are in:
```
scoring_results/intermediate_designs/aggregate_metrics_analyze.csv
```

Key metrics per nanobody:
| Metric | Meaning |
|---|---|
| `delta_sasa_refolded` | Buried surface area (higher = better binding) |
| `plip_hbonds_refolded` | H-bonds at interface |
| `largest_hydrophobic_refolded` | Hydrophobic patch size (lower = better) |
| `plddt` | Confidence score from folding model |

## Input file formats supported by generate_scoring_yamls.py

| Format | Example |
|---|---|
| FASTA | `filter_passed.fasta` — `>header` then sequence |
| CSV | `nanobodies.csv` — requires `sequence` column, optional `id` column |
| Plain text | `nanobodies.txt` — one sequence per line, `#` lines ignored |
