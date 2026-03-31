"""
Step 5 — RDKit Molecular Descriptors
======================================
Computes all available RDKit 2D descriptors for every ingredient that has
a valid SMILES string in pubchem_enriched.csv.

Input:
    outputs/pubchem_enriched.csv        (from step3)

Output:
    outputs/rdkit_descriptors.parquet
        index  : pubchem_cid
        columns: inci_name, canonical_smiles + all RDKit descriptor names

    outputs/rdkit_descriptors_meta.csv
        summary: cid, inci_name, n_valid_descriptors, n_nan_descriptors

Usage:
    python src/step5_rdkit_descriptors.py
    python src/step5_rdkit_descriptors.py --no-parquet   # write CSV instead
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
INPUT_CSV    = OUTPUT_DIR / "pubchem_enriched.csv"
OUTPUT_PQ    = OUTPUT_DIR / "rdkit_descriptors.parquet"
OUTPUT_CSV   = OUTPUT_DIR / "rdkit_descriptors.csv"
META_CSV     = OUTPUT_DIR / "rdkit_descriptors_meta.csv"

# RDKit descriptor names to exclude (non-numeric / deprecated / duplicate)
EXCLUDE_DESCRIPTORS = {
    "Ipc",          # often very large float, can cause overflow
}


def compute_rdkit_descriptors(smiles: str) -> dict[str, float] | None:
    """
    Compute all RDKit 2D descriptors for a SMILES string.
    Returns None if SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    result: dict[str, float] = {}
    for name, fn in Descriptors.descList:
        if name in EXCLUDE_DESCRIPTORS:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = fn(mol)
            result[name] = float(val) if val is not None else float("nan")
        except Exception:
            result[name] = float("nan")
    return result


def main(use_parquet: bool = True) -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run step3 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from step 3")

    # Only process rows with valid SMILES
    has_smiles = df[df["canonical_smiles"].notna() & (df["canonical_smiles"] != "")]
    no_smiles  = df[~(df["canonical_smiles"].notna() & (df["canonical_smiles"] != ""))]
    print(f"  {len(has_smiles)} rows with SMILES -> computing descriptors")
    print(f"  {len(no_smiles)} rows without SMILES -> skipped")

    # Deduplicate by SMILES (same compound may appear under different raw names)
    unique_smiles = has_smiles.drop_duplicates(subset=["canonical_smiles"])
    print(f"  {len(unique_smiles)} unique SMILES (after deduplication)")

    # Import RDKit once and get descriptor names
    from rdkit.Chem import Descriptors
    desc_names = [name for name, _ in Descriptors.descList if name not in EXCLUDE_DESCRIPTORS]
    print(f"  Computing {len(desc_names)} RDKit descriptors per molecule...")

    rows: list[dict] = []
    meta_rows: list[dict] = []
    n_invalid = 0

    for i, (_, row) in enumerate(unique_smiles.iterrows()):
        smiles   = row["canonical_smiles"]
        raw_name = row["inci_name"]
        inci     = row["inci_name"]
        cid      = row.get("pubchem_cid", "")

        descs = compute_rdkit_descriptors(smiles)

        if descs is None:
            n_invalid += 1
            print(f"  [{i+1:3d}] INVALID SMILES: {raw_name!r:<45} {smiles[:60]}")
            meta_rows.append({
                "pubchem_cid": cid,
                "inci_name": inci,
                "canonical_smiles": smiles,
                "n_valid_descriptors": 0,
                "n_nan_descriptors": len(desc_names),
                "smiles_valid": False,
            })
            continue

        n_valid = sum(1 for v in descs.values() if v == v)  # not NaN
        n_nan   = len(descs) - n_valid

        row_data = {
            "pubchem_cid":       cid,
            "inci_name":         inci,
            "canonical_smiles":  smiles,
            **descs,
        }
        rows.append(row_data)
        meta_rows.append({
            "pubchem_cid": cid,
            "inci_name": inci,
            "canonical_smiles": smiles,
            "n_valid_descriptors": n_valid,
            "n_nan_descriptors": n_nan,
            "smiles_valid": True,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(unique_smiles)}] processed...")

    print(f"\n  Done. {len(rows)} molecules computed, {n_invalid} invalid SMILES skipped.")

    # Build descriptor DataFrame
    desc_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta_rows)

    # Save outputs
    meta_df.to_csv(META_CSV, index=False)
    print(f"Wrote metadata -> {META_CSV}")

    if use_parquet:
        try:
            desc_df.to_parquet(OUTPUT_PQ, index=False)
            print(f"Wrote descriptors -> {OUTPUT_PQ}  ({OUTPUT_PQ.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"WARNING: parquet write failed ({e}), falling back to CSV")
            use_parquet = False

    if not use_parquet:
        desc_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote descriptors -> {OUTPUT_CSV}")

    # Summary statistics
    if rows:
        all_nan_rates = meta_df[meta_df["smiles_valid"]]["n_nan_descriptors"] / len(desc_names) * 100
        print(f"\n=== SUMMARY ===")
        print(f"  Unique SMILES processed    : {len(unique_smiles)}")
        print(f"  Valid molecules            : {len(rows)}")
        print(f"  Invalid SMILES             : {n_invalid}")
        print(f"  RDKit descriptors computed : {len(desc_names)}")
        print(f"  Avg NaN rate per molecule  : {all_nan_rates.mean():.1f}%")
        print(f"  Max NaN rate               : {all_nan_rates.max():.1f}%")
        print(f"  Molecules with 0% NaN      : {(all_nan_rates == 0).sum()}")

        # Per-descriptor NaN rate
        if len(rows) > 0:
            desc_only = desc_df[desc_names]
            col_nan_rates = desc_only.isna().mean() * 100
            high_nan = col_nan_rates[col_nan_rates > 20]
            print(f"  Descriptors with >20% NaN  : {len(high_nan)}")
            if len(high_nan) > 0:
                print("    Top offenders:")
                for col, rate in high_nan.sort_values(ascending=False).head(10).items():
                    print(f"      {col}: {rate:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5 — RDKit descriptors")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Write CSV instead of parquet")
    args = parser.parse_args()
    main(use_parquet=not args.no_parquet)
