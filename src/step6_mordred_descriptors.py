"""
Step 6 — Mordred 2D Molecular Descriptors
==========================================
Computes all 1613 Mordred 2D descriptors for every ingredient with a valid
SMILES string in pubchem_enriched.csv.

Input:
    outputs/pubchem_enriched.csv        (from step3)

Output:
    outputs/mordred_descriptors.parquet
        columns: pubchem_cid, inci_name, canonical_smiles + 1613 descriptor columns

    outputs/mordred_descriptors_meta.csv
        summary per molecule: valid/error counts, NaN rate

Usage:
    python src/step6_mordred_descriptors.py
    python src/step6_mordred_descriptors.py --no-parquet
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
OUTPUT_PQ    = OUTPUT_DIR / "mordred_descriptors.parquet"
OUTPUT_CSV   = OUTPUT_DIR / "mordred_descriptors.csv"
META_CSV     = OUTPUT_DIR / "mordred_descriptors_meta.csv"


def main(use_parquet: bool = True) -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run step3 first.", file=sys.stderr)
        sys.exit(1)

    from rdkit import Chem
    from mordred import Calculator, descriptors as mordred_descriptors

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from step 3")

    has_smiles = df[df["canonical_smiles"].notna() & (df["canonical_smiles"] != "")]
    print(f"  {len(has_smiles)} rows with SMILES")

    # Deduplicate by SMILES
    unique_smiles = has_smiles.drop_duplicates(subset=["canonical_smiles"])
    print(f"  {len(unique_smiles)} unique SMILES after deduplication")

    # Build Mordred calculator (2D only)
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    desc_names = [str(d) for d in calc.descriptors]
    print(f"  Mordred 2D descriptors: {len(desc_names)}")
    print("  Computing descriptors...")

    rows: list[dict] = []
    meta_rows: list[dict] = []
    n_invalid = 0

    for i, (_, row) in enumerate(unique_smiles.iterrows()):
        smiles   = row["canonical_smiles"]
        raw_name = row["inci_name"]
        inci     = row["inci_name"]
        cid      = row.get("pubchem_cid", "")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_invalid += 1
            print(f"  [{i+1:3d}] INVALID SMILES: {raw_name!r}")
            meta_rows.append({
                "pubchem_cid": cid,
                "inci_name": inci, "canonical_smiles": smiles,
                "n_computed": 0, "n_error": len(desc_names),
                "n_nan": len(desc_names), "smiles_valid": False,
            })
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calc(mol)

        desc_vals: dict[str, float] = {}
        n_err = 0
        for d, val in zip(calc.descriptors, result):
            try:
                fval = float(val)
                desc_vals[str(d)] = fval
            except (TypeError, ValueError):
                desc_vals[str(d)] = float("nan")
                n_err += 1

        n_nan = sum(1 for v in desc_vals.values() if v != v)
        n_ok  = len(desc_vals) - n_nan

        rows.append({
            "pubchem_cid":      cid,
            "inci_name":        inci,
            "canonical_smiles": smiles,
            **desc_vals,
        })
        meta_rows.append({
            "pubchem_cid": cid,
            "inci_name": inci, "canonical_smiles": smiles,
            "n_computed": n_ok, "n_error": n_err,
            "n_nan": n_nan, "smiles_valid": True,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(unique_smiles)}] processed...")

    print(f"\n  Done. {len(rows)} molecules, {n_invalid} invalid SMILES skipped.")

    desc_df  = pd.DataFrame(rows)
    meta_df  = pd.DataFrame(meta_rows)

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

    if rows:
        valid_meta = meta_df[meta_df["smiles_valid"]]
        nan_rates  = valid_meta["n_nan"] / len(desc_names) * 100
        print(f"\n=== SUMMARY ===")
        print(f"  Unique SMILES processed    : {len(unique_smiles)}")
        print(f"  Valid molecules            : {len(rows)}")
        print(f"  Invalid SMILES             : {n_invalid}")
        print(f"  Mordred descriptors        : {len(desc_names)}")
        print(f"  Avg NaN rate per molecule  : {nan_rates.mean():.1f}%")
        print(f"  Max NaN rate               : {nan_rates.max():.1f}%")
        print(f"  Molecules with 0% NaN      : {(nan_rates == 0).sum()}")

        if len(rows) > 0:
            col_nan_rates = desc_df[desc_names].isna().mean() * 100
            high_nan = col_nan_rates[col_nan_rates > 20]
            print(f"  Descriptors with >20% NaN  : {len(high_nan)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6 — Mordred 2D descriptors")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Write CSV instead of parquet")
    args = parser.parse_args()
    main(use_parquet=not args.no_parquet)
