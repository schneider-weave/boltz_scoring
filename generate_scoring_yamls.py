"""
Generate per-nanobody YAML input files for boltzgen scoring against P20809.

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

# ── Target config (validator: P20809 IL-11, clip_interval [36, 199]) ────────
# Full UniProt P20809:
# MNCVCRLVLVVLSLWPDTAVAPGPPPGPPRVSPDPRAELDSTVLLTRSLLADTRQLAAQLRDKFPADGDHNLDSLPTLAMSAGALGALQLPGVLTRLRADLLSYLRHVQWLRRAGGSSLKTLEPELGTLQARLDRLLRRLQLLMSRLALPQPPPDPPAPPLAPPSSAWGGIRAAHAILGGLHLTLDWAVRGLLLLKTRL
TARGET_ID = "P20809"
TARGET_CLIP_INTERVAL = (36, 199)  # 0-based, end exclusive — matches validator config.yaml
TARGET_SEQUENCE = (
    "AELDSTVLLTRSLLADTRQLAAQLRDKFPADGDHNLDSLPTLAMSAGALGALQLPGVLTRLRADLLSYLRHVQWLRRAGG"
    "SSLKTLEPELGTLQARLDRLLRRLQLLMSRLALPQPPPDPPAPPLAPPSSAWGGIRAAHAILGGLHLTLDWAVRGLLLLKTRL"
)
# MSA from https://github.com/metanova-labs/nova/tree/main/data/msa_files
MSA_PATH = f"../data/msa_files/{TARGET_ID}.a3m"

YAML_TEMPLATE = """\
entities:
- protein:
    id: A
    sequence: "{target_sequence}"
    msa: {msa_path}
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

    for nb_id, seq in sequences:
        content = YAML_TEMPLATE.format(
            target_sequence=TARGET_SEQUENCE,
            msa_path=MSA_PATH,
            nanobody_sequence=seq,
        )
        yaml_path = os.path.join(output_dir, f"{nb_id}.yaml")
        with open(yaml_path, "w") as f:
            f.write(content)

    print(f"Generated {len(sequences)} YAML files in: {output_dir}")
    print(f"Target: {TARGET_ID} ({len(TARGET_SEQUENCE)} aa, clip {TARGET_CLIP_INTERVAL})")
    print(f"MSA: {MSA_PATH}")
    print(f"\nNext step — run boltzgen scoring (validator-like):")
    print(f"  boltzgen run {output_dir} \\")
    print(f"      --output scoring_results/ \\")
    print(f"      --protocol nanobody-anything \\")
    print(f"      --skip_inverse_folding \\")
    print(f"      --num_designs 1 \\")
    print(f"      --steps design folding analysis \\")
    print(f"      --step_scale 2.0 \\")
    print(f"      --noise_scale 0.88")


def main():
    parser = argparse.ArgumentParser(
        description="Generate boltzgen scoring YAML files for nanobody sequences against P20809."
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
