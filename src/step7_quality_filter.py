"""
Step 7 — Quality Filter, Imputation & MLflow Logging
======================================================
Applies quality filtering to RDKit and Mordred descriptor matrices:
  1. Drop descriptor columns with NaN rate > NAN_THRESHOLD (default 20%)
  2. Median impute remaining NaN values
  3. Save versioned filtered outputs
  4. Log all parameters and metrics to MLflow

Inputs:
    outputs/rdkit_descriptors.parquet    (from step5)
    outputs/mordred_descriptors.parquet  (from step6)

Outputs:
    outputs/rdkit_filtered.parquet
    outputs/mordred_filtered.parquet
    outputs/quality_filter_report.csv    (per-column NaN rates before/after)

Usage:
    python src/step7_quality_filter.py
    python src/step7_quality_filter.py --nan-threshold 0.30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MLFLOW_DIR   = PROJECT_ROOT / "mlruns"

RDKIT_PQ     = OUTPUT_DIR / "rdkit_descriptors.parquet"
MORDRED_PQ   = OUTPUT_DIR / "mordred_descriptors.parquet"
RDKIT_OUT    = OUTPUT_DIR / "rdkit_filtered.parquet"
MORDRED_OUT  = OUTPUT_DIR / "mordred_filtered.parquet"
REPORT_CSV   = OUTPUT_DIR / "quality_filter_report.csv"

META_COLS = ["pubchem_cid", "inci_name", "canonical_smiles"]

NAN_THRESHOLD_DEFAULT = 0.20   # drop columns with >20% NaN


def filter_and_impute(
    df: pd.DataFrame,
    nan_threshold: float,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply NaN column filter + median imputation to descriptor DataFrame.
    Returns (filtered_df, metrics_dict).
    """
    # Separate meta columns from descriptor columns
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    desc_cols = [c for c in df.columns if c not in META_COLS]
    desc = df[desc_cols].copy().astype(float)

    n_rows_in = len(desc)
    n_cols_in = len(desc_cols)

    # ── Step 1: Drop columns with too many NaN ────────────────────────────────
    nan_rates = desc.isna().mean()
    cols_to_keep = nan_rates[nan_rates <= nan_threshold].index.tolist()
    cols_dropped = nan_rates[nan_rates > nan_threshold].index.tolist()

    desc_filtered = desc[cols_to_keep]
    n_cols_dropped = len(cols_dropped)
    n_cols_kept    = len(cols_to_keep)

    print(f"  [{label}] Columns in: {n_cols_in} | Dropped (>{nan_threshold*100:.0f}% NaN): "
          f"{n_cols_dropped} | Kept: {n_cols_kept}")

    # ── Step 2: Median imputation ─────────────────────────────────────────────
    nan_before_impute = desc_filtered.isna().sum().sum()
    medians = desc_filtered.median()
    desc_imputed = desc_filtered.fillna(medians)
    nan_after_impute = desc_imputed.isna().sum().sum()

    print(f"  [{label}] NaN before imputation: {nan_before_impute} | After: {nan_after_impute}")

    result = pd.concat([meta, desc_imputed], axis=1)

    metrics = {
        f"{label}_rows": n_rows_in,
        f"{label}_cols_in": n_cols_in,
        f"{label}_cols_dropped": n_cols_dropped,
        f"{label}_cols_kept": n_cols_kept,
        f"{label}_nan_before_impute": int(nan_before_impute),
        f"{label}_nan_after_impute": int(nan_after_impute),
        f"{label}_nan_threshold": nan_threshold,
    }

    return result, metrics, nan_rates, cols_to_keep


def main(nan_threshold: float = NAN_THRESHOLD_DEFAULT) -> None:
    # ── Check inputs ──────────────────────────────────────────────────────────
    missing = [p for p in [RDKIT_PQ, MORDRED_PQ] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: {p} not found. Run step5/step6 first.", file=sys.stderr)
        sys.exit(1)

    # ── Load descriptor matrices ──────────────────────────────────────────────
    print("Loading descriptor matrices...")
    rdkit_df   = pd.read_parquet(RDKIT_PQ)
    mordred_df = pd.read_parquet(MORDRED_PQ)
    print(f"  RDKit   : {rdkit_df.shape[0]} molecules x {rdkit_df.shape[1]} columns")
    print(f"  Mordred : {mordred_df.shape[0]} molecules x {mordred_df.shape[1]} columns")

    # ── Filter and impute ─────────────────────────────────────────────────────
    print(f"\nApplying NaN filter (threshold = {nan_threshold*100:.0f}%) + median imputation...")
    rdkit_filt,   rdkit_metrics,   rdkit_nan_rates,   rdkit_kept   = filter_and_impute(rdkit_df,   nan_threshold, "rdkit")
    mordred_filt, mordred_metrics, mordred_nan_rates, mordred_kept = filter_and_impute(mordred_df, nan_threshold, "mordred")

    all_metrics = {**rdkit_metrics, **mordred_metrics}

    # ── Save filtered outputs ─────────────────────────────────────────────────
    rdkit_filt.to_parquet(RDKIT_OUT, index=False)
    mordred_filt.to_parquet(MORDRED_OUT, index=False)
    print(f"\nWrote -> {RDKIT_OUT}   ({RDKIT_OUT.stat().st_size/1024:.1f} KB)")
    print(f"Wrote -> {MORDRED_OUT}  ({MORDRED_OUT.stat().st_size/1024:.1f} KB)")

    # ── Quality report CSV ────────────────────────────────────────────────────
    rdkit_report = pd.DataFrame({
        "descriptor": rdkit_nan_rates.index,
        "source": "rdkit",
        "nan_rate": rdkit_nan_rates.values,
        "kept": [c in rdkit_kept for c in rdkit_nan_rates.index],
    })
    mordred_report = pd.DataFrame({
        "descriptor": mordred_nan_rates.index,
        "source": "mordred",
        "nan_rate": mordred_nan_rates.values,
        "kept": [c in mordred_kept for c in mordred_nan_rates.index],
    })
    report = pd.concat([rdkit_report, mordred_report], ignore_index=True)
    report.to_csv(REPORT_CSV, index=False)
    print(f"Wrote -> {REPORT_CSV}")

    # ── MLflow logging ────────────────────────────────────────────────────────
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
        mlflow.set_experiment("phase0_quality_filter")

        with mlflow.start_run(run_name="step7_quality_filter"):
            mlflow.log_params({
                "nan_threshold": nan_threshold,
                "rdkit_cols_in": all_metrics["rdkit_cols_in"],
                "mordred_cols_in": all_metrics["mordred_cols_in"],
            })
            mlflow.log_metrics({k: v for k, v in all_metrics.items()
                                 if isinstance(v, (int, float))})
            mlflow.log_artifact(str(REPORT_CSV))
            mlflow.log_artifact(str(RDKIT_OUT))
            mlflow.log_artifact(str(MORDRED_OUT))
        print("Logged to MLflow.")
    except ImportError:
        print("WARNING: mlflow not installed, skipping experiment tracking.")
    except Exception as e:
        print(f"WARNING: MLflow logging failed: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n=== SUMMARY ===")
    print(f"  NaN threshold used         : {nan_threshold*100:.0f}%")
    print(f"  RDKit descriptors  : {all_metrics['rdkit_cols_in']:4d} in -> "
          f"{all_metrics['rdkit_cols_kept']:4d} kept  "
          f"({all_metrics['rdkit_cols_dropped']} dropped)")
    print(f"  Mordred descriptors: {all_metrics['mordred_cols_in']:4d} in -> "
          f"{all_metrics['mordred_cols_kept']:4d} kept  "
          f"({all_metrics['mordred_cols_dropped']} dropped)")
    print(f"  Molecules (both)   : {all_metrics['rdkit_rows']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7 — Quality filter + imputation")
    parser.add_argument("--nan-threshold", type=float, default=NAN_THRESHOLD_DEFAULT,
                        help=f"Drop columns with NaN rate above this (default {NAN_THRESHOLD_DEFAULT})")
    args = parser.parse_args()
    main(nan_threshold=args.nan_threshold)
