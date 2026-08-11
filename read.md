# Nanobody Scoring against P05231

Target setup matches the validator in [metanova-labs/nova](https://github.com/metanova-labs/nova)
(branch `inference-rework-and-structure-files`):

- Target: `P05231` (Interleukin-6, human), clip interval `[27, 212]`
- **Structure: `data/structures/4O9H.cif`, chain `A`, `res_index: 21..186`**
- **Epitope: `binding: 24,77,80,82,131,184..186`**
- MSA: `data/msa_files/P05231.a3m`

## The target is a structure now, not a sequence

The validator no longer passes IL-6 as a bare sequence. It passes **experimental
coordinates plus an explicit epitope**:

```yaml
entities:
- file:
    path: data/structures/4O9H.cif
    include:
        - chain:
            id: A
            res_index: 21..186
            msa: data/msa_files/P05231.a3m
    binding_types:
        - chain:
            id: A
            binding: 24,77,80,82,131,184..186
- protein:
    id: B
    sequence: "<design>"
    msa: empty
```

(`generate_scoring_yamls.py` writes this with absolute paths.)

Two consequences:

1. **boltzgen no longer folds IL-6** — it docks against the 4O9H backbone.
2. **`binding_types` pins where the nanobody binds.** Previously nothing did, so
   the diffusion sampler chose a different epitope on every draw. That produced
   swings of ~0.3 in `design_to_target_iptm` on *identical* input, which is what
   made local scores look unrelated to the validator's.

4O9H is IL-6 bound to an antibody Fab (chain A is IL-6; H/L are the Fab). The
listed residues are that antibody's epitope — designs are judged on hitting
**that** site, so designs screened before this change are not comparable.

> **Use nova's exact CIF.** `res_index` and `binding` are 1-based *positional*
> indices into the parsed chain (`boltzgen/data/parse/schema.py::parse_range`),
> **not** PDB author numbering. 4O9H entity 1 has exactly 186 SEQRES residues, so
> `21..186` is its last 166. A different copy of the structure shifts every index
> and silently targets the wrong site.
>
> ```bash
> mkdir -p data/structures
> curl -sSL -o data/structures/4O9H.cif \
>   https://raw.githubusercontent.com/metanova-labs/nova/inference-rework-and-structure-files/data/structures/4O9H.cif
> ```

Note the `.a3m` is carried for parity only: both this pipeline and the validator
build `Input(msa={})`, so the MSA never reaches the model and cannot move scores.

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

Requires `data/structures/4O9H.cif` to be present (see the curl command at the top);
the generator refuses to run without it, since a missing structure now means no
target at all rather than an unused file.

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

Keep about half the screened designs. (Earlier guidance here quoted specific
retention rates — ~32% for a top-10% cut, 75–83% for a 40–50% cut — derived from
the same discredited scatter estimate as §3b; they are removed, not restated.)
Stage 1 runs one replicate per sequence, so its ranking is a single noisy draw:
cut conservatively, and re-screen at `NUM_DESIGNS=3` if the `_sd` columns from
stage 2 show the spread is still wide.

`rm -rf finalists/` matters: boltzgen scores **every** YAML in the input
directory, so leftovers from a previous round would be scored again.

Design ids are `nb<line>_h<md5-of-sequence>`. The `nb0000` part renumbers when
the input file changes, but the `h...` hash is stable — use it to match a
stage-2 row back to its stage-1 row.

Run stage 2 as **one batch on one device**: `rank_sum` is relative to whichever
designs share the CSV, so it cannot be compared across separate runs or devices.

## 3b. How many replicates

The design and folding steps are diffusion samplers, so each run draws a different
structure and one run cannot order near-identical sequences.

**Re-measure the spread before choosing a replicate count.** An earlier version of
this file quoted `design_iiptm ±0.015` and claimed 3 replicates lift the chance of
picking the true best design from ~55% to ~69%. That scatter figure was measured
before the epitope was pinned and proved to be roughly 10× too small — one observed
pair of replicates on identical input differed by 0.29 in `design_to_target_iptm`.
Every probability derived from it is unreliable; the numbers have been removed
rather than restated.

Pinning the epitope (see the top of this file) removes the dominant source of that
variance, so the spread should now be far smaller — but it has not been measured
under the new spec. Run stage 2 at `NUM_DESIGNS=3`, look at the `_sd` columns, and
only raise the count if the spread is still comparable to the gaps you are trying
to resolve.

Read `validator_metrics.csv` as: **lowest `rank_sum` wins**, but two designs differ
meaningfully only when a metric gap exceeds roughly 3× its `_sd` column. Treat the top
few as a tied group rather than a strict ordering.

Note the validator scores each sequence **once**. No replicate count reproduces its
specific number — replicates estimate the centre of the distribution it draws from,
which is the most you can optimise for.

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
