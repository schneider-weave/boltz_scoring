#!/usr/bin/env python3
"""Export NOVA validator ranking metrics from boltzgen analyze CSV.

Reads aggregate_metrics_analyze.csv and writes a file with the 10 metrics
used for rank-sum scoring (boltzgen_config.yaml), plus optional local ranks.

Usage:
    python scripts/export_validator_metrics.py scoring_results/intermediate_designs/aggregate_metrics_analyze.csv
    python scripts/export_validator_metrics.py scoring_results/intermediate_designs/aggregate_metrics_analyze.csv -o validator_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
    wide = add_ranks(wide)

    rank_cols = [f"{m}_rank" for m in FLAT_METRICS]
    summary_cols = base_cols + list(FLAT_METRICS.keys()) + rank_cols + [
        "rank_sum",
        "worst_rank",
        "confidence_rank_sum",
        "physical_interaction_rank_sum",
        "developability_rank_sum",
        "validator_score",
    ]
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


if __name__ == "__main__":
    main()
