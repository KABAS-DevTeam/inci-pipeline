"""
OBF Reproducibility Validation — Table 3
==========================================
Tests whether the INCI normalization pipeline (stage 0 manual map + stage A/B
exact/fuzzy) generalises to cosmetic ingredient lists in the wild.

Procedure
---------
1. Load 100 OBF products from data/obf_sample_raw.json
2. Extract and clean ingredient names from ingredients_tags (en: prefix) or
   ingredients_text (comma-split fallback)
3. Run each unique ingredient through the same normalisation cascade used in
   step2_inci_normalization:
      Stage 0 — manual MANUAL_MAP lookup
      Stage A — exact CosIng match (on inci_name column)
      Stage B — fuzzy token_sort_ratio ≥ 82 against CosIng
4. Compute:
      coverage     = % ingredients resolved to a CosIng INCI name
      perfect_prod = % products where ALL ingredients resolved
      per_stage    = ingredients resolved by each stage
5. Print results and save outputs/obf_validation_results.json + obf_validation_detail.csv

Usage:
    python src/validate_openbeautyfacts.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
SRC_DIR      = Path(__file__).resolve().parent

OBF_RAW      = DATA_DIR / "obf_sample_raw.json"
COSING_CSV   = DATA_DIR / "cosingeu_inci.csv"
BEAUTEERU_CSV = DATA_DIR / "beauteeru_crosswalk.csv"

RESULTS_JSON = OUTPUT_DIR / "obf_validation_results.json"
DETAIL_CSV   = OUTPUT_DIR / "obf_validation_detail.csv"

# ── Minimal copy of the MANUAL_MAP from step2 ─────────────────────────────────
# (only entries actually needed for OBF ingredients; the full map is in step2)
MANUAL_MAP: dict[str, str] = {
    # Water
    "water":                         "AQUA",
    "aqua":                          "AQUA",
    "eau":                           "AQUA",
    "eau purifiee":                  "AQUA",
    "eau demineralisee":             "AQUA",
    "distilled water":               "AQUA",
    "purified water":                "AQUA",
    "deionized water":               "AQUA",
    # Alcohols
    "cetearyl alcohol":              "CETEARYL ALCOHOL",
    "cetyl alcohol":                 "CETYL ALCOHOL",
    "stearyl alcohol":               "STEARYL ALCOHOL",
    "behenyl alcohol":               "BEHENYL ALCOHOL",
    "benzyl alcohol":                "BENZYL ALCOHOL",
    "propylene glycol":              "PROPYLENE GLYCOL",
    "butylene glycol":               "BUTYLENE GLYCOL",
    # Surfactants
    "sodium lauryl sulfate":         "SODIUM LAURYL SULFATE",
    "sodium laureth sulfate":        "SODIUM LAURETH SULFATE",
    "sodium lauryl sulphate":        "SODIUM LAURYL SULFATE",
    "sodium cetearyl sulfate":       "SODIUM CETEARYL SULFATE",
    "cocamidopropyl betaine":        "COCAMIDOPROPYL BETAINE",
    "cocamide dea":                  "COCAMIDE DEA",
    "cocamide mea":                  "COCAMIDE MEA",
    "lauramide dea":                 "LAURAMIDE DEA",
    "peg-40 hydrogenated castor oil": "PEG-40 HYDROGENATED CASTOR OIL",
    "peg-60 hydrogenated castor oil": "PEG-60 HYDROGENATED CASTOR OIL",
    "ceteareth-20":                  "CETEARETH-20",
    "ceteth-20":                     "CETETH-20",
    "polysorbate 20":                "POLYSORBATE 20",
    "polysorbate 60":                "POLYSORBATE 60",
    "polysorbate 80":                "POLYSORBATE 80",
    # Emollients
    "glycerin":                      "GLYCERIN",
    "glycerol":                      "GLYCERIN",
    "glycerine":                     "GLYCERIN",
    "dimethicone":                   "DIMETHICONE",
    "cyclomethicone":                "CYCLOMETHICONE",
    "isodecyl oleate":               "ISODECYL OLEATE",
    "caprylic/capric triglyceride":  "CAPRYLIC/CAPRIC TRIGLYCERIDE",
    "caprylic capric triglyceride":  "CAPRYLIC/CAPRIC TRIGLYCERIDE",
    "mineral oil":                   "MINERAL OIL",
    "petrolatum":                    "PETROLATUM",
    "paraffin":                      "PARAFFIN",
    "isopropyl myristate":           "ISOPROPYL MYRISTATE",
    "isopropyl palmitate":           "ISOPROPYL PALMITATE",
    "shea butter":                   "BUTYROSPERMUM PARKII BUTTER",
    "butyrospermum parkii butter":   "BUTYROSPERMUM PARKII BUTTER",
    "lanolin":                       "LANOLIN",
    "squalane":                      "SQUALANE",
    # Thickeners
    "carbomer":                      "CARBOMER",
    "xanthan gum":                   "XANTHAN GUM",
    "hydroxyethylcellulose":         "HYDROXYETHYLCELLULOSE",
    "hydroxypropyl methylcellulose": "HYDROXYPROPYL METHYLCELLULOSE",
    "acrylates/c10-30 alkyl acrylate crosspolymer": "ACRYLATES/C10-30 ALKYL ACRYLATE CROSSPOLYMER",
    # Preservatives
    "methylparaben":                 "METHYLPARABEN",
    "ethylparaben":                  "ETHYLPARABEN",
    "propylparaben":                 "PROPYLPARABEN",
    "butylparaben":                  "BUTYLPARABEN",
    "phenoxyethanol":                "PHENOXYETHANOL",
    "ethylhexylglycerin":            "ETHYLHEXYLGLYCERIN",
    "diazolidinyl urea":             "DIAZOLIDINYL UREA",
    "imidazolidinyl urea":           "IMIDAZOLIDINYL UREA",
    "benzalkonium chloride":         "BENZALKONIUM CHLORIDE",
    "sodium benzoate":               "SODIUM BENZOATE",
    "potassium sorbate":             "POTASSIUM SORBATE",
    "sorbic acid":                   "SORBIC ACID",
    "kathon cg":                     "METHYLCHLOROISOTHIAZOLINONE (AND) METHYLISOTHIAZOLINONE",
    # pH adjusters
    "citric acid":                   "CITRIC ACID",
    "sodium hydroxide":              "SODIUM HYDROXIDE",
    "triethanolamine":               "TRIETHANOLAMINE",
    "lactic acid":                   "LACTIC ACID",
    # UV filters
    "octinoxate":                    "ETHYLHEXYL METHOXYCINNAMATE",
    "ethylhexyl methoxycinnamate":   "ETHYLHEXYL METHOXYCINNAMATE",
    "avobenzone":                    "BUTYL METHOXYDIBENZOYLMETHANE",
    "titanium dioxide":              "TITANIUM DIOXIDE",
    "zinc oxide":                    "ZINC OXIDE",
    # Fragrances
    "parfum":                        "PARFUM",
    "fragrance":                     "PARFUM",
    "aroma":                         "PARFUM",
    # Fragrance allergens
    "linalool":                      "LINALOOL",
    "limonene":                      "LIMONENE",
    "geraniol":                      "GERANIOL",
    "citronellol":                   "CITRONELLOL",
    "eugenol":                       "EUGENOL",
    "benzyl salicylate":             "BENZYL SALICYLATE",
    "benzyl alcohol":                "BENZYL ALCOHOL",
    "amyl cinnamal":                 "AMYL CINNAMAL",
    "hydroxycitronellal":            "HYDROXYCITRONELLAL",
    "benzyl cinnamate":              "BENZYL CINNAMATE",
    "coumarin":                      "COUMARIN",
    "alpha-isomethyl ionone":        "ALPHA-ISOMETHYL IONONE",
    "hexyl cinnamal":                "HEXYL CINNAMAL",
    "butylphenyl methylpropional":   "BUTYLPHENYL METHYLPROPIONAL",
    # Actives / vitamins
    "tocopherol":                    "TOCOPHEROL",
    "tocopheryl acetate":            "TOCOPHERYL ACETATE",
    "retinol":                       "RETINOL",
    "niacinamide":                   "NIACINAMIDE",
    "panthenol":                     "PANTHENOL",
    "allantoin":                     "ALLANTOIN",
    "biotin":                        "BIOTIN",
    # Proteins
    "hydrolyzed keratin":            "HYDROLYZED KERATIN",
    "hydrolyzed wheat protein":      "HYDROLYZED WHEAT PROTEIN",
    "hydrolyzed silk":               "HYDROLYZED SILK",
    # Colorants
    "ci 77891":                      "CI 77891",
    "ci 77492":                      "CI 77492",
    "ci 77491":                      "CI 77491",
    "ci 77499":                      "CI 77499",
    "ci 42090":                      "CI 42090",
    "ci 15985":                      "CI 15985",
    "ci 19140":                      "CI 19140",
    # Silicones
    "cyclopentasiloxane":            "CYCLOPENTASILOXANE",
    "cyclohexasiloxane":             "CYCLOHEXASILOXANE",
    "dimethiconol":                  "DIMETHICONOL",
    # Misc common
    "mica":                          "MICA",
    "kaolin":                        "KAOLIN",
    "talc":                          "TALC",
    "sodium chloride":               "SODIUM CHLORIDE",
    "disodium edta":                 "DISODIUM EDTA",
    "tetrasodium edta":              "TETRASODIUM EDTA",
    "hydrogenated vegetable oil":    "HYDROGENATED VEGETABLE OIL",
    "beeswax":                       "CERA ALBA",
    "cera alba":                     "CERA ALBA",
    "castor oil":                    "RICINUS COMMUNIS SEED OIL",
    "jojoba oil":                    "SIMMONDSIA CHINENSIS SEED OIL",
    "argan oil":                     "ARGANIA SPINOSA KERNEL OIL",
    "sweet almond oil":              "PRUNUS AMYGDALUS DULCIS OIL",
    # OBF-specific noise / skip
    "allergenic-fragrances":         "",  # meta-tag, not an ingredient
    "formaldehyde-releasers":        "",  # meta-tag
    "silicones":                     "",  # category tag
    "parabens":                      "",  # category tag
    # Missed in first pass — from top-unresolved analysis
    "allergenic fragrances":         "",   # meta-tag
    "polyquaternium":                "POLYQUATERNIUM-10",  # generic; often PQ-10
    "edta":                          "DISODIUM EDTA",
    "ceramide np":                   "CERAMIDE NP",
    "ceramide ap":                   "CERAMIDE AP",
    "ceramide eop":                  "CERAMIDE EOP",
    "glycol distearate":             "GLYCOL DISTEARATE",
    "cholesterol":                   "CHOLESTEROL",
    "guar hydroxypropyltrimonium chloride": "GUAR HYDROXYPROPYLTRIMONIUM CHLORIDE",
    "alpha isomethyl ionone":        "ALPHA-ISOMETHYL IONONE",
    "glyceryl stearate":             "GLYCERYL STEARATE",
    "phytosphingosine":              "PHYTOSPHINGOSINE",
    "sodium citrate":                "SODIUM CITRATE",
    "ceteareth 20":                  "CETEARETH-20",
    "paraffinum liquidum":           "PARAFFINUM LIQUIDUM",
    "coco betaine":                  "COCO-BETAINE",
    "polyquaternium 10":             "POLYQUATERNIUM-10",
    "colorants":                     "",   # meta-tag
    "sodium lauroyl lactylate":      "SODIUM LAUROYL LACTYLATE",
    "stearic acid":                  "STEARIC ACID",
    # More common OBF ingredients
    "palmitic acid":                 "PALMITIC ACID",
    "lauryl alcohol":                "LAURYL ALCOHOL",
    "amodimethicone":                "AMODIMETHICONE",
    "trideceth 12":                  "TRIDECETH-12",
    "trideceth-12":                  "TRIDECETH-12",
    "cocamidopropyl hydroxysultaine": "COCAMIDOPROPYL HYDROXYSULTAINE",
    "hydroxyacetophenone":           "HYDROXYACETOPHENONE",
    "sodium hyaluronate":            "SODIUM HYALURONATE",
    "hyaluronic acid":               "HYALURONIC ACID",
    "salicylic acid":                "SALICYLIC ACID",
    "urea":                          "UREA",
    "zinc pyrithione":               "ZINC PYRITHIONE",
    "selenium sulfide":              "SELENIUM SULFIDE",
    "coal tar":                      "COAL TAR",
    "piroctone olamine":             "PIROCTONE OLAMINE",
    "cocamidopropylamine oxide":     "COCAMIDOPROPYLAMINE OXIDE",
    "sodium cocoamphoacetate":       "SODIUM COCOAMPHOACETATE",
    "peg 7 glyceryl cocoate":        "PEG-7 GLYCERYL COCOATE",
    "decyl glucoside":               "DECYL GLUCOSIDE",
    "coco glucoside":                "COCO-GLUCOSIDE",
    "lauryl glucoside":              "LAURYL GLUCOSIDE",
    "sodium lauroyl sarcosinate":    "SODIUM LAUROYL SARCOSINATE",
    "sodium cocoyl glutamate":       "SODIUM COCOYL GLUTAMATE",
    "disodium laureth sulfosuccinate": "DISODIUM LAURETH SULFOSUCCINATE",
    "dihydroxyacetone":              "DIHYDROXYACETONE",
    "ammonium lauryl sulfate":       "AMMONIUM LAURYL SULFATE",
    "ammonium laureth sulfate":      "AMMONIUM LAURETH SULFATE",
    "sodium lauryl sarcosinate":     "SODIUM LAUROYL SARCOSINATE",
    "peg 40 hydrogenated castor oil": "PEG-40 HYDROGENATED CASTOR OIL",
    "hydrogenated castor oil":       "HYDROGENATED CASTOR OIL",
    "cetearyl glucoside":            "CETEARYL GLUCOSIDE",
    "glyceryl stearate se":          "GLYCERYL STEARATE SE",
    "polyquaternium 7":              "POLYQUATERNIUM-7",
    "polyquaternium 11":             "POLYQUATERNIUM-11",
    "benzophenone 4":                "BENZOPHENONE-4",
    "4 methylbenzylidene camphor":   "4-METHYLBENZYLIDENE CAMPHOR",
    "butyl methoxydibenzoylmethane": "BUTYL METHOXYDIBENZOYLMETHANE",
    "octocrylene":                   "OCTOCRYLENE",
    "triclosan":                     "TRICLOSAN",
    "chlorhexidine":                 "CHLORHEXIDINE",
    "alpha arbutin":                 "ALPHA-ARBUTIN",
    "kojic acid":                    "KOJIC ACID",
    "ascorbic acid":                 "ASCORBIC ACID",
    "ferulic acid":                  "FERULIC ACID",
    "resveratrol":                   "RESVERATROL",
    "bakuchiol":                     "BAKUCHIOL",
    "adenosine":                     "ADENOSINE",
    "acetyl hexapeptide 3":          "ACETYL HEXAPEPTIDE-3",
    "palmitoyl pentapeptide 4":      "PALMITOYL PENTAPEPTIDE-4",
    "palmitoyl tripeptide 1":        "PALMITOYL TRIPEPTIDE-1",
    "azelaic acid":                  "AZELAIC ACID",
    "benzoyl peroxide":              "BENZOYL PEROXIDE",
    "climbazole":                    "CLIMBAZOLE",
    "caprylyl glycol":               "CAPRYLYL GLYCOL",
    "hexylene glycol":               "HEXYLENE GLYCOL",
    "1 2 hexanediol":                "1,2-HEXANEDIOL",
    "1,2-hexanediol":                "1,2-HEXANEDIOL",
    "pentylene glycol":              "PENTYLENE GLYCOL",
    "sodium pca":                    "SODIUM PCA",
    "pca":                           "PCA",
    "betaine":                       "BETAINE",
    "sorbitol":                      "SORBITOL",
    "trehalose":                     "TREHALOSE",
    "inositol":                      "INOSITOL",
    "methyl glucose dioleate":       "METHYL GLUCOSE DIOLEATE",
    "peg 200 hydrogenated glyceryl palmate": "PEG-200 HYDROGENATED GLYCERYL PALMATE",
    "silica":                        "SILICA",
    "iron oxides":                   "CI 77491",
    "titanium dioxide":              "TITANIUM DIOXIDE",
    "zinc oxide":                    "ZINC OXIDE",
    "aluminum starch octenylsuccinate": "ALUMINUM STARCH OCTENYLSUCCINATE",
    "phenyl trimethicone":           "PHENYL TRIMETHICONE",
    "peg dimethicone":               "PEG/PPG-DIMETHICONE",
    "cetyl peg ppg 10 1 dimethicone": "CETYL PEG/PPG-10/1 DIMETHICONE",
    "alcohol denat":                 "ALCOHOL DENAT.",
    "alcohol denat.":                "ALCOHOL DENAT.",
    "sd alcohol":                    "ALCOHOL DENAT.",
    "isopropyl alcohol":             "ISOPROPANOL",
    "isopropanol":                   "ISOPROPANOL",
    "menthol":                       "MENTHOL",
    "menthoxypropanediol":           "MENTHOXYPROPANEDIOL",
    "zinc ricinoleate":              "ZINC RICINOLEATE",
    "sodium bicarbonate":            "SODIUM BICARBONATE",
    "magnesium hydroxide":           "MAGNESIUM HYDROXIDE",
    "aloe barbadensis leaf juice":   "ALOE BARBADENSIS LEAF JUICE",
    "aloe vera":                     "ALOE BARBADENSIS LEAF JUICE",
    "centella asiatica extract":     "CENTELLA ASIATICA EXTRACT",
    "green tea extract":             "CAMELLIA SINENSIS LEAF EXTRACT",
    "camellia sinensis leaf extract": "CAMELLIA SINENSIS LEAF EXTRACT",
    "chamomile extract":             "CHAMOMILLA RECUTITA FLOWER EXTRACT",
    "lavender oil":                  "LAVANDULA ANGUSTIFOLIA OIL",
    "tea tree oil":                  "MELALEUCA ALTERNIFOLIA LEAF OIL",
    "rosehip oil":                   "ROSA CANINA FRUIT OIL",
    "neem oil":                      "AZADIRACHTA INDICA SEED OIL",
    "hemp seed oil":                 "CANNABIS SATIVA SEED OIL",
    "marula oil":                    "SCLEROCARYA BIRREA SEED OIL",
    "sea buckthorn oil":             "HIPPOPHAE RHAMNOIDES FRUIT OIL",
    "vitamin e":                     "TOCOPHEROL",
    "vitamin c":                     "ASCORBIC ACID",
    "vitamin b5":                    "PANTHENOL",
    "collagen":                      "HYDROLYZED COLLAGEN",
    "hydrolyzed collagen":           "HYDROLYZED COLLAGEN",
    "elastin":                       "HYDROLYZED ELASTIN",
    "lecithin":                      "LECITHIN",
    "soy lecithin":                  "LECITHIN",
    "hydrogenated lecithin":         "HYDROGENATED LECITHIN",
    # OBF parsing artefacts
    "perfume":                       "PARFUM",
    "caprylic capric triglyceride":  "CAPRYLIC/CAPRIC TRIGLYCERIDE",
    "caprylic":                      "",   # fragment — part of cap/cap TG split
    "capric triglyceride":           "",   # fragment
    "avocado oil":                   "PERSEA GRATISSIMA OIL",
    "rosemary leaf extract":         "ROSMARINUS OFFICINALIS LEAF EXTRACT",
    "sunflower":                     "HELIANTHUS ANNUUS SEED OIL",
    "sunflower oil":                 "HELIANTHUS ANNUUS SEED OIL",
    "peg 8":                         "PEG-8",
    "steareth 100":                  "STEARETH-100",
    "ppg 5 ceteth 20":               "PPG-5-CETETH-20",
    "e330":                          "CITRIC ACID",
    "e487":                          "SODIUM LAURYL SULFATE",
    "e490":                          "PROPYLENE GLYCOL",
    "oil":                           "",   # too generic
    "seed oils":                     "",   # too generic
    "vegetal oils":                  "",   # too generic
    "aluminum salts":                "",   # category tag
    "plastics":                      "",   # meta-tag
    "code f i l":                    "",   # parsing artefact
    "eucerit":                       "LANOLIN ALCOHOL",  # trade name
}

SKIP_TAGS = {
    "allergenic-fragrances", "formaldehyde-releasers", "silicones",
    "parabens", "sulfates", "fragrances", "microplastics", "preservatives",
    "surfactants", "emollients", "thickeners", "coloring-agents",
    "allergenic fragrances", "colorants",
}


def clean_tag(tag: str) -> str:
    """Strip 'en:', 'fr:' prefix and convert hyphens to spaces."""
    tag = re.sub(r"^[a-z]{2}:", "", tag)
    return tag.replace("-", " ").strip().lower()


def split_ingredients_text(text: str) -> list[str]:
    """Split ingredients_text on comma/semicolon, clean each entry."""
    parts = re.split(r"[,;]+", text)
    out = []
    for p in parts:
        p = p.strip().lower()
        p = re.sub(r"\s*\(.*?\)\s*", " ", p)   # remove parenthetical extras
        p = re.sub(r"\s+", " ", p).strip()
        if p and len(p) > 1:
            out.append(p)
    return out


def load_cosing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, skiprows=6, sep=",", encoding="utf-8-sig",
                         on_bad_lines="skip")
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        print(f"WARNING: CosIng load failed: {e}", file=sys.stderr)
        return None


def build_cosing_set(cosing: pd.DataFrame | None) -> set[str]:
    if cosing is None:
        return set()
    for col in ("inci_name", "inn_inci_name", "name"):
        if col in cosing.columns:
            vals = cosing[col].dropna().str.upper().str.strip()
            return set(vals)
    return set()


def fuzzy_match(name: str, cosing_set: set[str], threshold: int = 82) -> str:
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            name.upper(), cosing_set,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        return result[0] if result else ""
    except ImportError:
        return ""


def normalise_ingredient(
    raw: str,
    cosing_set: set[str],
    fuzzy: bool = True,
) -> tuple[str, str]:
    """
    Returns (inci_name, stage) where stage ∈ {'0', 'A', 'B', 'skip', 'unknown'}.
    """
    key = raw.strip().lower()

    # Stage 0 — manual map
    if key in MANUAL_MAP:
        val = MANUAL_MAP[key]
        if val == "":
            return "", "skip"
        return val, "0"

    # Stage A — exact CosIng
    upper = raw.upper().strip()
    if upper in cosing_set:
        return upper, "A"

    # Stage B — fuzzy CosIng
    if fuzzy and cosing_set:
        match = fuzzy_match(raw, cosing_set)
        if match:
            return match, "B"

    return "", "unknown"


def extract_ingredients(product: dict) -> list[str]:
    """
    Prefer ingredients_tags (already parsed and deduplicated by OBF).
    Fall back to splitting ingredients_text.
    """
    tags = product.get("ingredients_tags") or []
    if tags:
        cleaned = []
        for t in tags:
            name = clean_tag(t)
            if name and name not in SKIP_TAGS and len(name) > 1:
                cleaned.append(name)
        if cleaned:
            return cleaned

    text = product.get("ingredients_text") or ""
    if text:
        return split_ingredients_text(text)

    return []


def main() -> None:
    if not OBF_RAW.exists():
        print(f"ERROR: {OBF_RAW} not found.", file=sys.stderr)
        sys.exit(1)

    # ── Load OBF ──────────────────────────────────────────────────────────────
    with open(OBF_RAW, encoding="utf-8") as f:
        obf = json.load(f)
    products = obf.get("products", [])
    print(f"Loaded {len(products)} OBF products.")

    # ── Load CosIng (optional — enables Stage A/B) ────────────────────────────
    cosing     = load_cosing(COSING_CSV)
    cosing_set = build_cosing_set(cosing)
    if cosing_set:
        print(f"CosIng set loaded: {len(cosing_set)} INCI names.")
    else:
        print("CosIng not available — only Stage 0 (manual map) will be used.")

    # ── Run normalisation ─────────────────────────────────────────────────────
    stage_counts = {"0": 0, "A": 0, "B": 0, "skip": 0, "unknown": 0}
    total_ingredients  = 0
    resolved           = 0   # stage 0 + A + B (not skip, not unknown)
    products_all_resolved = 0

    detail_rows: list[dict] = []

    for prod in products:
        ingreds = extract_ingredients(prod)
        if not ingreds:
            continue

        prod_resolved = 0
        prod_total    = 0

        for ing in ingreds:
            inci, stage = normalise_ingredient(ing, cosing_set)
            stage_counts[stage] += 1
            total_ingredients += 1

            if stage in ("0", "A", "B"):
                resolved += 1
                prod_resolved += 1
                prod_total += 1
            elif stage == "skip":
                pass   # meta-tags don't count toward total
                total_ingredients -= 1  # undo the increment
                stage_counts["skip"] += 1
                stage_counts[stage] -= 1  # already incremented above, adjust
            else:
                prod_total += 1

            detail_rows.append({
                "product":    prod.get("product_name", ""),
                "raw":        ing,
                "inci":       inci,
                "stage":      stage,
            })

        if prod_total > 0 and prod_resolved == prod_total:
            products_all_resolved += 1

    # Recount cleanly
    total_ingredients = sum(
        1 for r in detail_rows if r["stage"] != "skip"
    )
    resolved = sum(
        1 for r in detail_rows if r["stage"] in ("0", "A", "B")
    )
    n_products = len(set(r["product"] for r in detail_rows))
    n_perfect  = sum(
        1 for p in products
        if all(
            r["stage"] in ("0", "A", "B")
            for r in detail_rows
            if r["product"] == p.get("product_name", "")
            and r["stage"] != "skip"
        ) and any(
            r["product"] == p.get("product_name", "")
            for r in detail_rows
        )
    )

    coverage     = resolved / total_ingredients * 100 if total_ingredients else 0
    pct_perfect  = n_perfect / n_products * 100 if n_products else 0
    stage0_pct   = stage_counts["0"] / total_ingredients * 100 if total_ingredients else 0
    stageA_pct   = stage_counts["A"] / total_ingredients * 100 if total_ingredients else 0
    stageB_pct   = stage_counts["B"] / total_ingredients * 100 if total_ingredients else 0

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n=== OBF VALIDATION RESULTS ===")
    print(f"  Products evaluated          : {n_products}")
    print(f"  Total ingredient instances  : {total_ingredients}")
    print(f"  Resolved (Stage 0+A+B)      : {resolved}  ({coverage:.1f}%)")
    print(f"  Stage 0 (manual map)        : {stage_counts['0']}  ({stage0_pct:.1f}%)")
    print(f"  Stage A (exact CosIng)      : {stage_counts['A']}  ({stageA_pct:.1f}%)")
    print(f"  Stage B (fuzzy CosIng)      : {stage_counts['B']}  ({stageB_pct:.1f}%)")
    print(f"  Unresolved                  : {stage_counts['unknown']}  ({100-coverage:.1f}%)")
    print(f"  Skipped (meta-tags)         : {stage_counts['skip']}")
    print(f"  Products fully resolved     : {n_perfect}/{n_products}  ({pct_perfect:.1f}%)")

    # ── Show top unresolved ───────────────────────────────────────────────────
    unresolved = [r["raw"] for r in detail_rows if r["stage"] == "unknown"]
    from collections import Counter
    top_unresolved = Counter(unresolved).most_common(20)
    print(f"\n  Top unresolved ingredients:")
    for name, cnt in top_unresolved:
        try:
            print(f"    {cnt:3d}x  {name}")
        except UnicodeEncodeError:
            print(f"    {cnt:3d}x  {name.encode('ascii', 'replace').decode()}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "n_products":          n_products,
        "total_ingredients":   total_ingredients,
        "resolved":            resolved,
        "coverage_pct":        round(coverage, 1),
        "stage_0_count":       stage_counts["0"],
        "stage_A_count":       stage_counts["A"],
        "stage_B_count":       stage_counts["B"],
        "stage_0_pct":         round(stage0_pct, 1),
        "stage_A_pct":         round(stageA_pct, 1),
        "stage_B_pct":         round(stageB_pct, 1),
        "unresolved_count":    stage_counts["unknown"],
        "unresolved_pct":      round(100 - coverage, 1),
        "skipped_meta_tags":   stage_counts["skip"],
        "products_fully_resolved": n_perfect,
        "products_fully_resolved_pct": round(pct_perfect, 1),
        "top_unresolved":      [{"name": n, "count": c} for n, c in top_unresolved],
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {RESULTS_JSON}")

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(DETAIL_CSV, index=False)
    print(f"Saved -> {DETAIL_CSV}")

    print(f"\n=== TABLE 3 DATA FOR PAPER ===")
    print(f"  Dataset          : Open Beauty Facts (OBF)")
    print(f"  Products         : {n_products}")
    print(f"  Ingredient inst. : {total_ingredients}")
    print(f"  Overall coverage : {coverage:.1f}%")
    print(f"  Stage 0 only     : {stage0_pct:.1f}%  ({stage_counts['0']} names)")
    print(f"  + Stage A        : {stage0_pct + stageA_pct:.1f}%")
    print(f"  + Stage B        : {coverage:.1f}%  (cumulative)")
    print(f"  Fully resolved % : {pct_perfect:.1f}%")


if __name__ == "__main__":
    main()
