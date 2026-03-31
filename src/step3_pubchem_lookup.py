"""
Step 3 — PubChem SMILES Lookup
================================
Retrieves Canonical SMILES (and metadata) from PubChem PUG REST API
for each INCI-normalized ingredient.

Priority order for each ingredient:
  1. Use existing pubchem_cid from step 2 (manual map or beauteeru)
  2. Search by CAS number → get CID → get SMILES
  3. Search by INCI name → get CID → get SMILES
  4. Mark as SMILES_NOT_FOUND

Rate limiting: 5 requests/second (PubChem public API limit).

Input:
    outputs/inci_normalized.csv        (from step2)

Output:
    outputs/pubchem_enriched.csv
        columns: inci_name, cas_number, pubchem_cid,
                 canonical_smiles, iupac_name, molecular_formula,
                 molecular_weight, smiles_source, match_method, confidence

Usage:
    python src/step3_pubchem_lookup.py
    python src/step3_pubchem_lookup.py --dry-run    # show what would be looked up
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
INPUT_CSV    = OUTPUT_DIR / "inci_normalized.csv"
OUTPUT_CSV   = OUTPUT_DIR / "pubchem_enriched.csv"

# ── PubChem API constants ─────────────────────────────────────────────────────
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_DELAY = 0.22   # 4.5 req/sec to stay under 5/sec limit

# INCI names that are mixtures / polymers / fragrances — PubChem won't have them
SKIP_INCI_PATTERNS = {
    "PARFUM",
    "PRESERVATIVE (UNSPECIFIED)",
    "ANTIMICROBIAL AGENT (UNSPECIFIED)",
    "CORROSION INHIBITOR (UNSPECIFIED)",
    "ANTIFREEZE (UNSPECIFIED)",
    "SOLUBILIZER (UNSPECIFIED)",
    "STABILIZER (UNSPECIFIED)",
    "SEQUESTERING AGENT (UNSPECIFIED)",
    "OPACIFIER (UNSPECIFIED)",
    "COLORANT (UNSPECIFIED)",
    "PROPELLANT (UNSPECIFIED)",
    "SURFACTANT BLEND",
    "DETERGENT BASE (UNSPECIFIED)",
    "NONIONIC SURFACTANT (UNSPECIFIED)",
    "PHOSPHONATE COMPOUND",
    "PEPPERMINT FLAVOR",
    "TOOTHPASTE FLAVOR",
    "NACREOUS PEARLESCENT PIGMENT",
    "WATER/SOLVENT",
}


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """GET JSON from PubChem with retry on 503/429."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited (HTTP {r.status_code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            return None
        except requests.RequestException as e:
            print(f"    Request error: {e}")
            time.sleep(1)
    return None


def cid_from_cas(cas: str) -> str | None:
    """Look up PubChem CID by CAS number."""
    # Handle multi-CAS strings (e.g. "3088-31-1 / 9004-82-4 / ...")
    cas_clean = cas.split("/")[0].strip()
    if not cas_clean:
        return None
    url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(cas_clean)}/cids/JSON"
    data = _get(url)
    if data and "IdentifierList" in data:
        cids = data["IdentifierList"].get("CID", [])
        if cids:
            return str(cids[0])
    time.sleep(REQUEST_DELAY)
    return None


def cid_from_name(inci_name: str) -> str | None:
    """Look up PubChem CID by INCI name (or synonym)."""
    if not inci_name or inci_name in SKIP_INCI_PATTERNS:
        return None
    # Try exact name match first
    url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(inci_name)}/cids/JSON"
    data = _get(url)
    if data and "IdentifierList" in data:
        cids = data["IdentifierList"].get("CID", [])
        if cids:
            time.sleep(REQUEST_DELAY)
            return str(cids[0])
    time.sleep(REQUEST_DELAY)
    return None


def _clean_cid(cid: str) -> str:
    """Normalize CID to integer string (handles '753.0' from pandas float)."""
    try:
        return str(int(float(cid)))
    except (ValueError, TypeError):
        return ""


def properties_from_cid(cid: str) -> dict:
    """Retrieve SMILES, IUPAC name, formula, MW from PubChem CID."""
    cid_int = _clean_cid(cid)
    if not cid_int or cid_int == "0":
        return {}
    # PubChem returns ConnectivitySMILES (not CanonicalSMILES) for this endpoint
    props = "ConnectivitySMILES,IsomericSMILES,IUPACName,MolecularFormula,MolecularWeight"
    url = f"{PUBCHEM_BASE}/compound/cid/{cid_int}/property/{props}/JSON"
    data = _get(url)
    time.sleep(REQUEST_DELAY)
    if data and "PropertyTable" in data:
        table = data["PropertyTable"].get("Properties", [])
        if table:
            p = table[0]
            # Prefer IsomericSMILES, fall back to ConnectivitySMILES
            smiles = p.get("IsomericSMILES") or p.get("ConnectivitySMILES", "")
            return {
                "pubchem_cid": str(p.get("CID", cid_int)),
                "canonical_smiles": smiles,
                "iupac_name": p.get("IUPACName", ""),
                "molecular_formula": p.get("MolecularFormula", ""),
                "molecular_weight": str(p.get("MolecularWeight", "")),
            }
    return {}


def lookup_smiles(row: dict, dry_run: bool = False) -> dict:
    """
    Try all strategies to get SMILES for one ingredient row.
    Returns updated row dict with SMILES fields added.
    """
    result = dict(row)
    result.setdefault("canonical_smiles", "")
    result.setdefault("iupac_name", "")
    result.setdefault("molecular_formula", "")
    result.setdefault("molecular_weight", "")
    result.setdefault("smiles_source", "")

    inci = str(row.get("inci_name", "")).strip().upper()
    cas  = str(row.get("cas_number", "")).strip()
    cid  = str(row.get("pubchem_cid", "")).strip()

    # Skip non-chemical entries
    if row.get("match_method") in ("skipped_solvent", "skipped_noise"):
        result["smiles_source"] = "skipped"
        return result

    # Skip known mixtures / fragrances
    inci_base = inci.split("(")[0].strip()
    if any(inci_base.startswith(p) or inci == p for p in SKIP_INCI_PATTERNS):
        result["smiles_source"] = "not_applicable"
        return result

    if dry_run:
        result["smiles_source"] = "dry_run"
        return result

    # Strategy 1: Use existing CID from step 2
    cid_clean = _clean_cid(cid)
    if cid_clean and cid_clean != "0":
        props = properties_from_cid(cid_clean)
        if props.get("canonical_smiles"):
            result.update(props)
            result["smiles_source"] = "cid_from_step2"
            return result

    # Strategy 2: CAS → CID → SMILES
    if cas and cas not in ("", "nan"):
        new_cid = cid_from_cas(cas)
        if new_cid:
            props = properties_from_cid(new_cid)
            if props.get("canonical_smiles"):
                result.update(props)
                result["smiles_source"] = "cas_lookup"
                return result

    # Strategy 3: INCI name → CID → SMILES
    if inci and inci not in SKIP_INCI_PATTERNS:
        new_cid = cid_from_name(inci)
        if new_cid:
            props = properties_from_cid(new_cid)
            if props.get("canonical_smiles"):
                result.update(props)
                result["smiles_source"] = "inci_name_lookup"
                return result

    result["smiles_source"] = "not_found"
    return result


def main(dry_run: bool = False) -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run step2 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    print(f"Loaded {len(df)} ingredients from step 2")

    # Count what needs lookup
    skip_count = df[df["match_method"].isin(["skipped_solvent", "skipped_noise"])].shape[0]
    lookup_needed = len(df) - skip_count
    print(f"  {skip_count} water/noise rows to skip")
    print(f"  {lookup_needed} rows to query PubChem")

    if dry_run:
        print("\nDRY RUN — no actual HTTP requests")

    fields = [
        "inci_name", "cas_number", "pubchem_cid",
        "canonical_smiles", "iupac_name", "molecular_formula", "molecular_weight",
        "smiles_source", "match_method", "confidence", "notes",
    ]

    results: list[dict] = []
    n_smiles = 0
    n_not_found = 0
    n_skipped = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        updated = lookup_smiles(row_dict, dry_run=dry_run)
        results.append(updated)

        source = updated.get("smiles_source", "")
        has_smiles = bool(updated.get("canonical_smiles"))

        if source in ("skipped", "not_applicable"):
            n_skipped += 1
            status = "skip"
        elif has_smiles:
            n_smiles += 1
            status = f"OK ({source})"
        else:
            n_not_found += 1
            status = "NOT FOUND"

        if idx % 10 == 0 or source == "not_found":
            raw = row_dict.get("inci_name", "")[:50]
            print(f"  [{idx+1:3d}/{len(df)}] {raw!r:<52} {status}")

    # Write output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows -> {OUTPUT_CSV}")
    print(f"\n=== SUMMARY ===")
    print(f"  Total processed    : {len(df)}")
    print(f"  SMILES retrieved   : {n_smiles}")
    print(f"  Not found          : {n_not_found}")
    print(f"  Skipped (water/mix): {n_skipped}")

    lookable = len(df) - n_skipped
    if lookable > 0:
        print(f"  SMILES coverage    : {n_smiles}/{lookable} = {n_smiles/lookable*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3 — PubChem SMILES lookup")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be queried without making API calls")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
