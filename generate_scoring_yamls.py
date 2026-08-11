"""
Generate per-nanobody YAML input files for boltzgen scoring against P05231.

Input file formats supported:
  - FASTA  (.fasta / .fa)  — headers used as IDs, e.g. >design_spec_0673|rank=4
  - CSV    (.csv)          — requires a 'sequence' column; optional 'id' column
  - Plain text             — one sequence per line, '#' lines ignored

Usage:
    python generate_scoring_yamls.py --input filter_passed.fasta --output_dir scoring_inputs/
    python generate_scoring_yamls.py --input nanobodies.csv      --output_dir scoring_inputs/
    python generate_scoring_yamls.py --input nanobodies.txt      --output_dir scoring_inputs/

Then run boltzgen on the whole directory:
    boltzgen run scoring_inputs/ \
        --output scoring_results/ \
        --protocol nanobody-anything \
        --skip_inverse_folding \
        --num_designs 1
"""

import argparse
import os
import csv
import hashlib
from pathlib import Path

# ── Target config — mirrors nova config.yaml protein_selection.nanobody ──
# The target is now supplied as an experimental STRUCTURE, not a bare sequence:
# boltzgen docks against 4O9H chain A coordinates instead of folding IL-6 itself,
# and binding_types pins the epitope so the pose is no longer sampled freely.
#
# res_index / binding are 1-based POSITIONAL indices into the parsed chain
# (see boltzgen/data/parse/schema.py::parse_range) — NOT PDB auth numbering.
# 4O9H entity 1 has exactly 186 SEQRES residues, so 21..186 is its last 166.
# This is why the CIF must be nova's byte-identical copy: a different file
# shifts every index and silently targets the wrong epitope.
TARGET_ID = "P05231"
TARGET_CLIP_INTERVAL = (27, 212)  # kept for reference; the structure now defines the target
STRUCTURE_ID = "4O9H"
STRUCTURE_CHAIN = "A"
STRUCTURE_RES_INDEX = "21..186"
STRUCTURE_BINDING_SITE = "24,77,80,82,131,184..186"

# Absolute paths: boltzgen resolves `file.path` itself, and a stale relative
# path would previously have gone unnoticed because the MSA was never loaded.
_REPO_ROOT = Path(__file__).resolve().parent
# Structure + MSA from https://github.com/metanova-labs/nova (branch:
# inference-rework-and-structure-files) — data/structures and data/msa_files.
STRUCTURE_PATH = _REPO_ROOT / "data" / "structures" / f"{STRUCTURE_ID}.cif"
MSA_PATH = _REPO_ROOT / "data" / "msa_files" / f"{TARGET_ID}.a3m"

YAML_TEMPLATE = """\
entities:
- file:
    path: {structure_path}
    include:
        - chain:
            id: {structure_chain}
            res_index: {structure_res_index}
            msa: {msa_path}
    binding_types:
        - chain:
            id: {structure_chain}
            binding: {structure_binding_site}
- protein:
    id: B
    sequence: "{nanobody_sequence}"
    msa: empty
"""


def seq_id(sequence: str, idx: int) -> str:
    """Generate a short deterministic ID from the sequence hash."""
    h = hashlib.md5(sequence.encode()).hexdigest()[:8]
    return f"nb{idx:04d}_h{h}"


def _sanitize_id(raw: str) -> str:
    """Strip '>' and replace characters that are invalid in filenames."""
    return raw.lstrip(">").strip().replace("/", "_").replace(" ", "_").replace("|", "_")


def load_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into (id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq: list[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq).upper()))
                current_id = _sanitize_id(line[1:])
                current_seq = []
            else:
                current_seq.append(line)

    # flush last record
    if current_id is not None and current_seq:
        sequences.append((current_id, "".join(current_seq).upper()))

    return sequences


def load_sequences(input_path: str) -> list[tuple[str, str]]:
    """
    Returns a list of (id, sequence) tuples.
    Auto-detects FASTA, CSV, or plain-text format.
    """
    path = Path(input_path)
    sequences = []

    # ── FASTA ──────────────────────────────────────────────────────────────────
    if path.suffix.lower() in (".fasta", ".fa", ".faa"):
        return load_fasta(path)

    with open(path, "r") as f:
        sample = f.read(2048)
        f.seek(0)

        # Also detect FASTA by content (first non-empty line starts with '>')
        first_line = next((l.strip() for l in sample.splitlines() if l.strip()), "")
        if first_line.startswith(">"):
            return load_fasta(path)

        # ── CSV ────────────────────────────────────────────────────────────────
        if "," in sample or "\t" in sample:
            dialect = "excel-tab" if "\t" in sample else "excel"
            reader = csv.DictReader(f, dialect=dialect)
            headers = [h.strip().lower() for h in (reader.fieldnames or [])]

            seq_col = next(
                (h for h in headers if h in ("sequence", "seq", "nanobody_sequence", "aa_sequence")),
                None,
            )
            if seq_col is None:
                raise ValueError(
                    f"Could not find a sequence column in CSV. "
                    f"Expected one of: sequence, seq, nanobody_sequence, aa_sequence. "
                    f"Found: {headers}"
                )

            id_col = next((h for h in headers if h in ("id", "name", "nanobody_id")), None)

            for i, row in enumerate(reader):
                row = {k.strip().lower(): v.strip() for k, v in row.items()}
                seq = row[seq_col].strip().upper()
                if not seq:
                    continue
                nb_id = row[id_col].strip() if id_col and row.get(id_col) else seq_id(seq, i)
                sequences.append((nb_id, seq))

        # ── Plain text ─────────────────────────────────────────────────────────
        else:
            for i, line in enumerate(f):
                seq = line.strip().upper()
                if not seq or seq.startswith("#"):
                    continue
                sequences.append((seq_id(seq, i), seq))

    return sequences


def generate_yamls(sequences: list[tuple[str, str]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Fail loudly here rather than one-by-one inside boltzgen: a missing CIF now
    # means no target at all, not just a silently unused file.
    if not STRUCTURE_PATH.exists():
        raise FileNotFoundError(
            f"Target structure not found: {STRUCTURE_PATH}\n"
            f"Fetch nova's exact copy (do not substitute an RCSB download):\n"
            f"  mkdir -p {STRUCTURE_PATH.parent}\n"
            f"  curl -sSL -o {STRUCTURE_PATH} https://raw.githubusercontent.com/"
            f"metanova-labs/nova/inference-rework-and-structure-files/"
            f"data/structures/{STRUCTURE_ID}.cif"
        )
    if not MSA_PATH.exists():
        raise FileNotFoundError(f"Target MSA not found: {MSA_PATH}")

    for nb_id, seq in sequences:
        content = YAML_TEMPLATE.format(
            structure_path=STRUCTURE_PATH,
            structure_chain=STRUCTURE_CHAIN,
            structure_res_index=STRUCTURE_RES_INDEX,
            structure_binding_site=STRUCTURE_BINDING_SITE,
            msa_path=MSA_PATH,
            nanobody_sequence=seq,
        )
        yaml_path = os.path.join(output_dir, f"{nb_id}.yaml")
        with open(yaml_path, "w") as f:
            f.write(content)

    print(f"Generated {len(sequences)} YAML files in: {output_dir}")
    print(f"Target:    {TARGET_ID} (clip {TARGET_CLIP_INTERVAL})")
    print(f"Structure: {STRUCTURE_ID} chain {STRUCTURE_CHAIN}, res_index {STRUCTURE_RES_INDEX}")
    print(f"Epitope:   {STRUCTURE_BINDING_SITE}")
    print(f"MSA:       {MSA_PATH}")
    print(f"\nNext step — full scoring pipeline:")
    print(f"  CACHE=/workspace/cache bash scripts/run_scoring.sh {output_dir} scoring_results/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate boltzgen scoring YAML files for nanobody sequences against P05231."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help=(
            "Input file containing nanobody sequences. Supported formats:\n"
            "  FASTA (.fasta/.fa)  — e.g. filter_passed.fasta\n"
            "  CSV                 — requires a 'sequence' column; optional 'id' column\n"
            "  Plain text          — one sequence per line, '#' lines ignored"
        ),
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="scoring_inputs",
        help="Directory to write YAML files into. Default: scoring_inputs/",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    sequences = load_sequences(args.input)
    if not sequences:
        raise ValueError("No sequences found in input file.")

    print(f"Loaded {len(sequences)} nanobody sequences from {args.input}")
    generate_yamls(sequences, args.output_dir)


if __name__ == "__main__":
    main()
