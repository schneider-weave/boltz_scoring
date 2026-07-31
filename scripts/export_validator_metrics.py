#!/usr/bin/env python3
"""Export NOVA validator ranking metrics from boltzgen analyze CSV.

Reads aggregate_metrics_analyze.csv and writes a file with the 10 metrics
used for rank-sum scoring (boltzgen_config.yaml), plus optional local ranks.

Scoring is stochastic: the diffusion design and folding steps draw fresh noise
every run, so a single replicate carries as much scatter as the real spread
between similar designs. With --num_designs N > 1 boltzgen emits N replicates
per input as `<stem>_0 .. <stem>_<N-1>`; those are collapsed here to a median
per sequence, with `<metric>_sd` reporting the standard error of that median.

Usage:
    python scripts/export_validator_metrics.py scoring_results/intermediate_designs/aggregate_metrics_analyze.csv
    python scripts/export_validator_metrics.py scoring_results/intermediate_designs/aggregate_metrics_analyze.csv -o validator_scores.csv
    python scripts/export_validator_metrics.py ... --no-aggregate   # keep every replicate
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# `<stem>_<k>` suffix boltzgen appends when more than one design per input is written.
REPLICATE_SUFFIX = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)$")

# Matches https://github.com/metanova-labs/nova/blob/main/config/boltzgen_config.yaml
VALIDATOR_METRICS: dict[str, dict[str, str]] = {
    "confidence": {
        "design_iiptm": "max",
        "design_ptm": "max",
        "design_to_target_iptm": "max",
        "min_design_to_target_pae": "min",
        "interaction_pae": "min",
    },
    "physical_interaction": {
        "plip_hbonds_refolded": "max",
        "plip_saltbridge_refolded": "max",
        "delta_sasa_refolded": "max",
    },
    "developability": {
        "liability_score": "min",
        "liability_num_violations": "min",
    },
}

FLAT_METRICS = {
    metric: mode
    for metrics in VALIDATOR_METRICS.values()
    for metric, mode in metrics.items()
}


def metric_category(metric: str) -> str:
    for category, metrics in VALIDATOR_METRICS.items():
        if metric in metrics:
            return category
    raise KeyError(metric)


def split_replicate_id(sample_id: str) -> tuple[str, int | None]:
    """Split `nb0000_habc12345_2` into ('nb0000_habc12345', 2)."""
    match = REPLICATE_SUFFIX.match(str(sample_id))
    if match is None:
        return str(sample_id), None
    return match.group("stem"), int(match.group("idx"))


def looks_like_replicates(df: pd.DataFrame) -> bool:
    """True only when ids are `<stem>_<k>` with k running 0..n-1 within every stem.

    Guards against input YAMLs whose own names end in a number (`design_1.yaml`),
    which would otherwise be merged into a single bogus group.
    """
    split = df["id"].map(split_replicate_id)
    if any(idx is None for _, idx in split):
        return False
    groups: dict[str, set[int]] = {}
    for stem, idx in split:
        groups.setdefault(stem, set()).add(idx)
    if all(len(indices) == 1 for indices in groups.values()):
        return False
    return all(indices == set(range(len(indices))) for indices in groups.values())


def aggregate_replicates(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """Collapse replicate rows to one row per sequence: median + standard error.

    `<metric>` becomes the median across replicates and `<metric>_sd` the
    standard error of that median (per-replicate SD / sqrt(n)). Ranking on the
    median is what makes the ordering reproducible; `_sd` says how far apart two
    designs must sit before the difference means anything (roughly 3x).
    """
    out = df.copy()
    out["stem"] = out["id"].map(lambda s: split_replicate_id(s)[0])

    rows = []
    for stem, group in out.groupby("stem", sort=False):
        row: dict[str, object] = {"id": stem, "n_replicates": len(group)}
        for col in base_cols:
            if col != "id":
                row[col] = group[col].iloc[0]
        for metric in FLAT_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = values.median()
            row[f"{metric}_sd"] = (
                values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric, mode in FLAT_METRICS.items():
        ascending = mode == "min"
        out[f"{metric}_rank"] = out[metric].rank(method="dense", ascending=ascending)

    rank_cols = [f"{m}_rank" for m in FLAT_METRICS]
    out["rank_sum"] = out[rank_cols].sum(axis=1)
    out["worst_rank"] = out[rank_cols].max(axis=1)

    for category, metrics in VALIDATOR_METRICS.items():
        cat_rank_cols = [f"{m}_rank" for m in metrics]
        out[f"{category}_rank_sum"] = out[cat_rank_cols].sum(axis=1)

    out["validator_score"] = out["rank_sum"]
    return out.sort_values("rank_sum", ascending=True)


def export_long(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        for metric, mode in FLAT_METRICS.items():
            rows.append(
                {
                    "id": row["id"],
                    "category": metric_category(metric),
                    "metric": metric,
                    "mode": mode,
                    "score": row[metric],
                    "rank": row.get(f"{metric}_rank"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NOVA validator ranking metrics from analyze CSV."
    )
    parser.add_argument(
        "input_csv",
        help="Path to aggregate_metrics_analyze.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="validator_metrics.csv",
        help="Wide-format output CSV (default: validator_metrics.csv)",
    )
    parser.add_argument(
        "--long-output",
        default=None,
        help="Optional long-format CSV (one row per metric per design)",
    )
    parser.add_argument(
        "--include-sequence",
        action="store_true",
        help="Include designed_sequence column in wide output",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Keep one row per replicate instead of collapsing to a median per sequence",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    raw = pd.read_csv(input_path)
    missing = [m for m in FLAT_METRICS if m not in raw.columns]
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {', '.join(missing)}")

    base_cols = ["id"]
    if "designed_sequence" in raw.columns and args.include_sequence:
        base_cols.append("designed_sequence")

    wide = raw[base_cols + list(FLAT_METRICS.keys())].copy()

    aggregated = not args.no_aggregate and looks_like_replicates(wide)
    if aggregated:
        replicates = len(wide)
        wide = aggregate_replicates(wide, base_cols)
        print(
            f"Aggregated {replicates} replicate rows -> {len(wide)} sequences "
            f"(median per metric, {int(wide['n_replicates'].min())}-"
            f"{int(wide['n_replicates'].max())} replicates each)"
        )
    elif not args.no_aggregate:
        print(
            "Single replicate per sequence — scores carry full run-to-run scatter. "
            "Re-run with --num_designs 3 for a stable ranking."
        )

    wide = add_ranks(wide)

    rank_cols = [f"{m}_rank" for m in FLAT_METRICS]
    sd_cols = [f"{m}_sd" for m in FLAT_METRICS] if aggregated else []
    count_cols = ["n_replicates"] if aggregated else []
    summary_cols = (
        base_cols
        + count_cols
        + list(FLAT_METRICS.keys())
        + sd_cols
        + rank_cols
        + [
            "rank_sum",
            "worst_rank",
            "confidence_rank_sum",
            "physical_interaction_rank_sum",
            "developability_rank_sum",
            "validator_score",
        ]
    )
    wide[summary_cols].to_csv(args.output, index=False)
    print(f"Wrote {len(wide)} designs -> {args.output}")

    if args.long_output:
        export_long(wide, Path(args.long_output))
        print(f"Wrote long format -> {args.long_output}")

    best = wide.iloc[0]
    print(
        f"Best local rank_sum: {best['rank_sum']:.0f} "
        f"(worst_rank={best['worst_rank']:.0f}, id={best['id']})"
    )
    if aggregated and len(wide) > 1:
        print("Top candidates (lowest rank_sum wins):")
        for _, row in wide.head(3).iterrows():
            print(f"  {row['id']:<24s} rank_sum={row['rank_sum']:.0f}")
        print(
            "  Designs whose rank_sum differs by only a few points are tied; "
            "a metric gap must exceed ~3x its _sd column to be real."
        )


if __name__ == "__main__":
    main()
