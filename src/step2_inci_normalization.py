"""
Step 2 — INCI Normalization
============================
Maps raw French ingredient names to standard INCI names using a 3-stage cascade:

  Stage A — Exact string match against CosIng EU + beauteeru crosswalk
  Stage B — Fuzzy match using token_sort_ratio (threshold 82)
  Stage C — Claude API (claude-sonnet-4-6) for remaining unresolved names

Input:
    outputs/raw_ingredient_names.csv          (from step1)

Reference data:
    data/cosingeu_inci.csv                    (24,094 INCI entries)
    data/beauteeru_crosswalk.csv              (28,354 INCI→CAS→PubChem_CID rows)

Output:
    outputs/inci_normalized.csv
        columns: raw_name, inci_name, cas_number, pubchem_cid,
                 match_method, match_score, confidence, notes

Usage:
    python src/step2_inci_normalization.py
    python src/step2_inci_normalization.py --skip-llm   # skip Claude API calls
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

# ── Optional imports (warn if missing) ────────────────────────────────────────
try:
    from rapidfuzz import fuzz, process as rfprocess
    FUZZY_LIB = "rapidfuzz"
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        FUZZY_LIB = "fuzzywuzzy"
    except ImportError:
        FUZZY_LIB = None

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "virtual_lab" / "phase0_data_foundation" / "data"
OUTPUT_DIR   = PROJECT_ROOT / "virtual_lab" / "phase0_data_foundation" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV    = OUTPUT_DIR / "raw_ingredient_names.csv"
OUTPUT_CSV   = OUTPUT_DIR / "inci_normalized.csv"
COSING_CSV   = DATA_DIR / "cosingeu_inci.csv"
BEAUTEERU_CSV = DATA_DIR / "beauteeru_crosswalk.csv"

FUZZY_THRESHOLD = 82   # token_sort_ratio threshold (0–100); tune empirically

# ── Non-ingredient filter — skip these raw names entirely ─────────────────────
SKIP_NAMES = {
    "eau traitée", "eau déminéralisée", "eau", "water", "eau traitee",
    "qsp 100", "qsp", "total", "q.s.p.", "eau purifiée",
    # Noise: company names and document metadata
    "enad shymeca", "shymeca", "enad", "e'n a'd", "en a d",
    "designation:", "designation",
    # Noise: usage instructions
    "ne pas avaler ; en cas d'ingestion",
    "utilisé pour la fabrication de savon liquide pour main.",
    "utilise pour la fabrication de shampooing, eaux de cologne ; liquide vaisselle ; after shave",
    "utilisé pour la fabrication des after shave et des pates dentifrices",
    "utilisé pour les produits cosmétiques et l'hygiène corporelle en tant qu'agent antibactérien et antifongique",
    # Noise: multi-ingredient composition strings (FICHE TECHNIQUE PRODUIT FINI)
    "eau déminéralisée tensio - actif non ionique base neutralisante parfum colorant agent de conservation",
    "eau déminéralisée agent nettoyage agent brillant de cuir solvant alcool agent de conservation",
    "tensio - actif anionique et non ionique solvant colorant agent de conservation eau déminéralisée séquestrant anti rouille",
    "tensio - actifs anionique et non ionique solvant agent dégraissant alcalin agent de conservation séquestrant",
    "tensio - actifs anionique et non ionique solvant organique agent dégraissant alcalin agent de conservation séquestrant",
    "eau déminéralisée soude caustique tensio - actif non ionique et amphotère agent de conservation",
    "tensio actifs anionique et non ionique biodégradable à 90 % agent mouillant agent dégraissant agent de conservation eau",
    "tensio - actifs anionique et non ionique solvant agent de conservation stabilisant anti- rouille parfum",
    "tensio - actifs non ionique solvants base dégraissante agent de conservation colorant eau déminéralisée",
    "tensio-actif anionique et non ionique solvant colorant agent de conservation eau déminéralisée séquestrant anti rouille",
    "tensioactif non-ionique acide sulfonique acide chlorhydrique. parfum eau.",
    "agent humectant tensio-actif non ionique parfum opacifiant agent de conservation eau déminéralisée",
    "agent nettoyeur de plastique agent brillant de plastique parfum agent de conservation eau déminéralisée",
    "anionique et non ionique colorant parfum agent de conservation savon anti mousse viscosant",
    "hypochlorite de sodium eau déminéralisée tensio - actifs anioniques et amphotères. base",
    "eau déminéralisée alcool iso propylique glycol alkyl phénol base neutralisante formol colorant.",
    "eau déminéralisée. tensio - actifs",
    "eau déminéralisée. tensio - actif anionique colorant parfum agent de conservation charge minérale acide gras",
    "eau déminéralisée. tensio – actifs anionique base dégraissante agent humectant colorant",
    "eau déminéralisée tensio-actif amphotère sequestrant parfum colorant agent de conservation",
    "eau déminéralisée tensio-actif cationique. colorant parfum agent de conservation",
    "eau déminéralisée. tensio-actif non ionique parfum. opacifiant colorant. conservateur",
    "eau déminéralisée. colorants agent de conservation cires solvant emulsifiant",
    "eau déminéralisée. acide chlorhydrique.",
    # Long document fragments from HTML table extraction noise
    "eau déminéralisée. colorants agent de conservation cires solvant emulsifiant  # specifications: aspect : pâte visqueuse couleur : noir",
    "teneur en eau?3",
    "teneur en eau≤3",
    # Spec noise — moisture content ranges, not ingredient names
    "4.5<teneur en eau≤5",
    "3<teneur en eau<4.5",
    "agent anti gel agent anti corrosif minéral colorant eau déminéralisée",
    # Noise: long HTML/document fragments
    "**utilisation:**<br>",
    "**shymeca**",
    # Noise: pure water entries
    "eau déminalisée",
    "eau déminéralisée.",
    "eau traitée.",
    "eau déminéralisée est obtenue à partir de l'eau brute traitée à l'aide de station osmose inverse.",
    "eau déminéralisée. colorants agent de conservation cires solvant emulsifiant",
}

# ── Known manual mappings for common abbreviations / trade names ──────────────
# Format: raw_name_lower → (inci_name, cas, pubchem_cid, notes)
MANUAL_MAP: dict[str, tuple[str, str, str, str]] = {
    "ap9":                    ("NONOXYNOL-9", "9016-45-9", "24756940", "Nonylphenol ethoxylate 9 EO"),
    "ap 9":                   ("NONOXYNOL-9", "9016-45-9", "24756940", "Nonylphenol ethoxylate 9 EO"),
    "les na":                 ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", "LES-Na 70% solution"),
    "les 70%":                ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", "LES-Na 70% solution"),
    "les":                    ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", ""),
    "l.e.s na à 70%":         ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", ""),
    "bg":                     ("BUTOXYETHANOL", "111-76-2", "8057", "Butyl glycol / Butyl cellosolve"),
    "butyle glycol":          ("BUTOXYETHANOL", "111-76-2", "8057", "Butyl glycol"),
    "mpg":                    ("PROPYLENE GLYCOL", "57-55-6", "1030", "Mono Propylene Glycol"),
    "ddbsh":                  ("SODIUM DODECYLBENZENESULFONATE", "25155-30-0", "23665752", "DDBSH = dodecylbenzene sulfonate"),
    "acide sulfonique (ddbsh)":("SODIUM DODECYLBENZENESULFONATE", "25155-30-0", "23665752", ""),
    "acide sulfonique ddbsh":  ("SODIUM DODECYLBENZENESULFONATE", "25155-30-0", "23665752", ""),
    "betaine":                ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", "Assumed CAPB; confirm"),
    "bétaïne":                ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", "Assumed CAPB; confirm"),
    "formol":                 ("FORMALDEHYDE", "50-00-0", "712", "Formol = 37% formaldehyde solution"),
    "naoh":                   ("SODIUM HYDROXIDE", "1310-73-2", "14798", ""),
    "naoh pur":               ("SODIUM HYDROXIDE", "1310-73-2", "14798", ""),
    "nacl":                   ("SODIUM CHLORIDE", "7647-14-5", "5234", ""),
    "sel industriel nacl":    ("SODIUM CHLORIDE", "7647-14-5", "5234", ""),
    "edta":                   ("TETRASODIUM EDTA", "64-02-8", "33040", "Tetrasodium EDTA assumed; could be disodium"),
    "e.d.t.a":                ("TETRASODIUM EDTA", "64-02-8", "33040", ""),
    "ap7":                    ("NONOXYNOL-7", "9016-45-9", "24756940", "Nonylphenol ethoxylate 7 EO"),
    "menthol":                ("MENTHOL", "89-78-1", "16666", ""),
    "glycerine":              ("GLYCERIN", "56-81-5", "753", ""),
    "glycérine":              ("GLYCERIN", "56-81-5", "753", ""),
    "glycerine codex":        ("GLYCERIN", "56-81-5", "753", "Pharmaceutical grade"),
    "glycérine codex":        ("GLYCERIN", "56-81-5", "753", ""),
    "sorbitol":               ("SORBITOL", "50-70-4", "5780", ""),
    "paraffine":              ("PARAFFIN", "8002-74-2", "6922266", ""),
    "talc":                   ("TALC", "14807-96-6", "24961609", ""),
    "lanoline":               ("LANOLIN", "8006-54-0", "11970101", ""),
    "carbomer 980 ou 940":    ("CARBOMER", "9003-01-4", "2723949", "Crosslinked polyacrylic acid"),
    "carbomer":               ("CARBOMER", "9003-01-4", "2723949", ""),
    "irgasan dp 300":         ("TRICLOSAN", "3380-34-5", "5564", "Trade name Irgasan DP300"),
    "irgasan dp300":          ("TRICLOSAN", "3380-34-5", "5564", ""),
    "lorol c16":              ("CETYL ALCOHOL", "36653-82-4", "2682496", "Lorol C16 = cetyl alcohol"),
    "nigrosine s.":           ("CI 50415", "8005-03-6", "3084073", "Nigrosine dye"),
    "blanc de titane":        ("TITANIUM DIOXIDE", "13463-67-7", "26042", ""),
    "monofluorophosphate de sodium":("SODIUM MONOFLUOROPHOSPHATE", "10163-15-2", "24948", ""),
    "fluorure de sodium":     ("SODIUM FLUORIDE", "7681-49-4", "5284567", ""),
    "acide chlorhydrique concentré":("HYDROCHLORIC ACID", "7647-01-0", "313", ""),
    "acide chlorhydrique concentré à 30%":("HYDROCHLORIC ACID", "7647-01-0", "313", "30% solution"),
    "acide chlorhydrique centre":("HYDROCHLORIC ACID", "7647-01-0", "313", "OCR artifact of 'concentré'"),
    "hydroxyde de calcium":   ("CALCIUM HYDROXIDE", "1305-62-0", "6093208", ""),
    "carbonate de sodium":    ("SODIUM CARBONATE", "497-19-8", "10340", ""),
    "carbonate de calcium":   ("CALCIUM CARBONATE", "471-34-1", "10112", ""),
    "phosphate trisodique":   ("TRISODIUM PHOSPHATE", "7601-54-9", "24455", ""),
    "meta silicate de sodium": ("SODIUM METASILICATE", "6834-92-0", "23792", ""),
    "meta silicatéde sodium":  ("SODIUM METASILICATE", "6834-92-0", "23792", "OCR typo"),
    "métasilicate de sodium":  ("SODIUM METASILICATE", "6834-92-0", "23792", ""),
    "silicate de soude":       ("SODIUM SILICATE", "1344-09-8", "3423151", ""),
    "myristate d'isopropyle":  ("ISOPROPYL MYRISTATE", "110-27-0", "8042", ""),
    "beurre de karité":        ("BUTYROSPERMUM PARKII BUTTER", "91080-23-8", "5366085", "Shea butter"),
    "alcool éthylique à 96°":  ("ALCOHOL DENAT.", "64-17-5", "702", "96% ethanol"),
    "alcool éthylique pharma": ("ALCOHOL", "64-17-5", "702", "Pharmaceutical ethanol"),
    "alcool éthylique dénaturé":("ALCOHOL DENAT.", "64-17-5", "702", ""),
    "alcool à 96°":            ("ALCOHOL DENAT.", "64-17-5", "702", ""),
    "propylène glycol":        ("PROPYLENE GLYCOL", "57-55-6", "1030", ""),
    "mono éthylène glycol inhibé":("ETYLENE GLYCOL", "107-21-1", "174", "Inhibited monoethylene glycol (antifreeze)"),
    "d-panthéanol":            ("PANTHENOL", "81-13-0", "131204", "D-Panthenol = Provitamin B5"),
    "d. panthenol":            ("PANTHENOL", "81-13-0", "131204", ""),
    "triéthanolamine à 85 %":  ("TRIETHANOLAMINE", "102-71-6", "7618", "TEA 85%"),
    "tea 85":                  ("TRIETHANOLAMINE", "102-71-6", "7618", "TEA 85%"),
    "carboxymethylcellulose":  ("CELLULOSE GUM", "9000-11-7", "6420601", "CMC"),
    "carboxyméthylcellulose":  ("CELLULOSE GUM", "9000-11-7", "6420601", "CMC"),
    "white spirit":            ("HYDROCARBONS, C9-C12, N-ALKANES, ISOALKANES, CYCLICS", "64742-47-8", "516308", "White spirit solvent"),
    "potasse caustique":       ("POTASSIUM HYDROXIDE", "1310-58-3", "14797", "KOH"),
    "soude caustique":         ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH"),
    "soude caustique naoh":    ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH"),
    "acide stéarique d.p":     ("STEARIC ACID", "57-11-4", "5281", "Double Pression grade"),
    "acide stéarique dp":      ("STEARIC ACID", "57-11-4", "5281", "Double Pression grade"),
    "acide stéarique lp":      ("STEARIC ACID", "57-11-4", "5281", ""),
    "acide gras de coco":      ("COCONUT ACID", "67701-05-7", "5365679", "Mixed coconut fatty acids"),
    "chlorure de potassium":   ("POTASSIUM CHLORIDE", "7447-40-7", "4873", ""),
    "anti mousse":             ("DIMETHICONE", "9006-65-9", "24765", "Antifoam = silicone dimethicone assumed"),

    # ── Additional trade names and French synonyms ─────────────────────────────
    # Alcohol variants
    "alcool éthylique":        ("ALCOHOL DENAT.", "64-17-5", "702", ""),
    "alcool ethylique":        ("ALCOHOL DENAT.", "64-17-5", "702", ""),
    "alcool isopropylique":    ("ISOPROPYL ALCOHOL", "67-63-0", "3776", "IPA"),
    "alcool iso propylique":   ("ISOPROPYL ALCOHOL", "67-63-0", "3776", "IPA"),

    # Glycol antifreeze variants
    "mono ethylene glycol inhibé":         ("ETHYLENE GLYCOL", "107-21-1", "174", "Inhibited MEG antifreeze"),
    "mono ethylene glycol meg inhibé":     ("ETHYLENE GLYCOL", "107-21-1", "174", "Inhibited MEG"),
    "mono ethylene glycol inhibé oat (additif 100% organique g12)": ("ETHYLENE GLYCOL", "107-21-1", "174", "OAT inhibited MEG G12"),
    "mono ethylene glycol inhibé oat (additif 100% organique g13)": ("ETHYLENE GLYCOL", "107-21-1", "174", "OAT inhibited MEG G13"),
    "ethylène glycol inhibé minéral eau déminéralisée": ("ETHYLENE GLYCOL", "107-21-1", "174", "Mineral inhibited glycol blend"),

    # Preservatives
    "kathon cg":               ("METHYLCHLOROISOTHIAZOLINONE (AND) METHYLISOTHIAZOLINONE", "26172-55-4", "57482", "Kathon CG"),
    "kathan cg":               ("METHYLCHLOROISOTHIAZOLINONE (AND) METHYLISOTHIAZOLINONE", "26172-55-4", "57482", "Kathon CG variant"),
    "igrasan dp 300":          ("TRICLOSAN", "3380-34-5", "5564", "Irgasan DP300"),
    "igrasan dp300":           ("TRICLOSAN", "3380-34-5", "5564", ""),
    "parahydroxybenzoate de méthyle": ("METHYLPARABEN", "99-76-3", "7456", ""),
    "parahydroxybenzoate de methyle": ("METHYLPARABEN", "99-76-3", "7456", ""),
    "parahydroxybenzoate de propyle": ("PROPYLPARABEN", "94-13-3", "7175", ""),

    # Surfactants — trade names
    "texapon n70":             ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", "Texapon N70 = SLES 70%"),
    "texapon n 70":            ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", ""),
    "empicol lx 95":           ("SODIUM LAURYL SULFATE", "151-21-3", "3423265", "Empicol LX 95 = SLS 95%"),
    "lauryl ether sulfate de sodium (texapon n70)": ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", ""),
    "lauryl ether sulfate of sodium (texapon n70)": ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", ""),
    "laryyl éther sulfate de sodium": ("SODIUM LAURETH SULFATE", "9004-82-4", "23665763", "OCR typo of lauryl"),
    "comperlan cod":           ("COCAMIDE DEA", "61791-31-9", "8053426", "Comperlan COD = Cocamide DEA"),
    "comperlan":               ("COCAMIDE DEA", "61791-31-9", "8053426", "Assumed Comperlan COD"),
    "comperian":               ("COCAMIDE DEA", "61791-31-9", "8053426", "OCR typo of Comperlan"),
    "cocamide dea (comperlan cod)": ("COCAMIDE DEA", "61791-31-9", "8053426", ""),
    "coco betaine":            ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "coco-betaine":            ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "coco-bétaïne":            ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "coco-betaine (dehyton ab 30)": ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "coco-bétaïne (dehyton ab 30)": ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "cocoamidopropyl betaine /coco-betaine (dehyton ab 30)": ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "dehyton ab30":            ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", "Dehyton AB30 = CAPB 30%"),
    "dehyton ab 30":           ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", ""),
    "plantacare 818":          ("COCO-GLUCOSIDE", "141464-42-8", "16136216", "Plantacare 818 = Coco-glucoside"),
    "coco glucoside et oléate de glycérile (lamesoft po 65)": ("COCO-GLUCOSIDE (AND) GLYCERYL OLEATE", "141464-42-8", "16136216", "Lamesoft PO 65"),
    "coco glucoside et oléate de glycérile": ("COCO-GLUCOSIDE (AND) GLYCERYL OLEATE", "141464-42-8", "16136216", ""),
    "coco glucoside et oléate de glycéryle": ("COCO-GLUCOSIDE (AND) GLYCERYL OLEATE", "141464-42-8", "16136216", ""),
    "lamesoft po 65":          ("COCO-GLUCOSIDE (AND) GLYCERYL OLEATE", "141464-42-8", "16136216", ""),
    "lamesoft p065":           ("COCO-GLUCOSIDE (AND) GLYCERYL OLEATE", "141464-42-8", "16136216", ""),
    "lamesoft po65":           ("COCO-GLUCOSIDE (AND) GLYCERYL OLEATE", "141464-42-8", "16136216", ""),

    # Conditioning agents
    "dehyquart a":             ("QUATERNIUM-18", "61789-80-8", "9570420", "Dehyquart A = quaternary ammonium surfactant"),
    "dehyquat a":              ("QUATERNIUM-18", "61789-80-8", "9570420", "OCR typo of Dehyquart A"),
    "merquat 550":             ("POLYQUATERNIUM-7", "26590-05-6", "16219946", "Merquat 550"),
    "merquat 550 cc7":         ("POLYQUATERNIUM-7", "26590-05-6", "16219946", ""),
    "polymer pq 10":           ("POLYQUATERNIUM-10", "68610-92-4", "16217556", "PQ-10 = hydroxyethyl cellulose quat"),
    "polymer pq10":            ("POLYQUATERNIUM-10", "68610-92-4", "16217556", ""),
    "luvimer 100p":            ("ACRYLATES COPOLYMER", "25133-97-5", "15939647", "Luvimer 100P = acrylates copolymer"),

    # Emulsifiers and fatty alcohols
    "lanette o":               ("CETEARYL ALCOHOL", "67762-27-0", "8936", "Lanette O = cetearyl alcohol"),
    "laneette o":              ("CETEARYL ALCOHOL", "67762-27-0", "8936", "OCR typo Lanette O"),
    "lanoxal 75":              ("CETEARETH-25", "68439-49-6", "23674612", "Lanoxal 75 = ceteareth emulsifier"),
    "emulgin b1":              ("CETEARETH-12", "68439-49-6", "24765", "Emulgin B1 = ceteareth-12"),
    "emulgine b1":             ("CETEARETH-12", "68439-49-6", "24765", "Emulgin B1 variant"),
    "cutina ags":              ("GLYCERYL STEARATE (AND) PEG-100 STEARATE", "31566-31-1", "5283468", "Cutina AGS self-emulsifying base"),

    # Silicones
    "emulsion de silicone bc 2268":   ("DIMETHICONE", "9006-65-9", "24765", "Silicone emulsion (trade name redacted)"),
    "emulsion silicone bc 2268":      ("DIMETHICONE", "9006-65-9", "24765", "Silicone emulsion (trade name redacted)"),
    "emulsion de silicone bc 91/023": ("DIMETHICONE", "9006-65-9", "24765", "Silicone emulsion 60% (trade name redacted)"),
    "emulsion silicone bc 91/023":    ("DIMETHICONE", "9006-65-9", "24765", "Silicone emulsion (trade name redacted)"),
    "bc 91/023 (émulsion siliconnée 60%)": ("DIMETHICONE", "9006-65-9", "24765", "Silicone emulsion 60% (trade name redacted)"),

    # Panthenol variants
    "d-panthénol":             ("PANTHENOL", "81-13-0", "131204", "D-Panthenol"),
    "d-panténol":              ("PANTHENOL", "81-13-0", "131204", "OCR variant"),

    # Dyes / Colorants
    "colorant jaune tartrazine": ("CI 19140", "1934-21-0", "164825", "Tartrazine / FD&C Yellow 5"),
    "colorant jaune tartazine":  ("CI 19140", "1934-21-0", "164825", "OCR typo"),
    "dioxyde de titane":         ("TITANIUM DIOXIDE", "13463-67-7", "26042", ""),
    "noir de carbone":           ("CI 77266", "1333-86-4", "5462329", "Carbon black"),
    "azurant optique":           ("DISODIUM DISTYRYLBIPHENYL DISULFONATE", "27344-41-8", "25151880", "Optical brightener"),
    "azurant optique liquide":   ("DISODIUM DISTYRYLBIPHENYL DISULFONATE", "27344-41-8", "25151880", ""),
    "azurant optique o.b en poudre": ("DISTYRYLBIPHENYL DISULFONATE", "27344-41-8", "25151880", "OB optical brightener powder"),

    # Abrasives / Toothpaste
    "sident 22s":              ("HYDRATED SILICA", "7699-41-4", "6124190", "Sident 22S abrasive silica"),
    "sident 9":                ("HYDRATED SILICA", "7699-41-4", "6124190", "Sident 9 thickening silica"),
    "silice abrasif":          ("HYDRATED SILICA", "7699-41-4", "6124190", ""),
    "alumine hydratée":        ("ALUMINA", "1344-28-1", "9989226", "Al2O3·3H2O abrasive"),
    "mono fluoro phosphate de sodium": ("SODIUM MONOFLUOROPHOSPHATE", "10163-15-2", "24948", ""),
    "arôme menthe poivrée":    ("PEPPERMINT FLAVOR", "", "", "Flavor ingredient — no single CAS"),
    "arome menthe poivree":    ("PEPPERMINT FLAVOR", "", "", ""),
    "arôme bi fluor":          ("TOOTHPASTE FLAVOR", "", "", "Proprietary dental flavor"),
    "arome pour dentifrice bi-fluor": ("TOOTHPASTE FLAVOR", "", "", ""),

    # Shea butter variant
    "beurre de karite":        ("BUTYROSPERMUM PARKII BUTTER", "91080-23-8", "5366085", "Shea butter"),

    # Acids
    "formaldéhyde":            ("FORMALDEHYDE", "50-00-0", "712", ""),
    "peroxyde d'hydrogène":    ("HYDROGEN PEROXIDE", "7722-84-1", "763", ""),
    "acide sulfurique":        ("SULFURIC ACID", "7664-93-9", "1118", ""),
    "phosphonates":            ("PHOSPHONATE COMPOUND", "", "", "Generic phosphonate class — CAS depends on specific compound"),
    "borax":                   ("SODIUM TETRABORATE", "1303-96-4", "10219853", ""),
    "acide citrique":          ("CITRIC ACID", "77-92-9", "311", ""),
    "acide citrique.":         ("CITRIC ACID", "77-92-9", "311", ""),
    "butyl hydroxy anisol":    ("BHA", "25013-16-5", "8456", "Butylhydroxyanisole antioxidant"),
    "carboxy méthyl cellulose": ("CELLULOSE GUM", "9000-11-7", "6420601", "CMC"),
    "carboxy méthylcellulose":  ("CELLULOSE GUM", "9000-11-7", "6420601", "CMC"),
    "carboxymethyle cellulose de sodium": ("CELLULOSE GUM", "9000-11-7", "6420601", "CMC"),
    "butyl-glycol":            ("BUTOXYETHANOL", "111-76-2", "8057", "Butyl glycol"),
    "amp 95":                  ("AMP (AMINOMETHYL PROPANOL)", "124-68-5", "8572", "AMP-95 = 2-amino-2-methyl-1-propanol 95%"),
    "soudé caustique":         ("SODIUM HYDROXIDE", "1310-73-2", "14798", "OCR typo of soude caustique"),
    "hypochlorite concentré":  ("SODIUM HYPOCHLORITE", "7681-52-9", "23665763", "Concentrated NaOCl"),
    "emulcool 64% (v/v)":      ("ALCOHOL DENAT.", "64-17-5", "702", "Emulcool 64% = denatured alcohol base"),
    "emulcool 64.5%":          ("ALCOHOL DENAT.", "64-17-5", "702", ""),
    "cetiol he":               ("PEG-7 GLYCERYL COCOATE", "68201-46-7", "25152395", "Cetiol HE = PEG-7 glyceryl cocoate"),
    "tenzaryl":                ("BENZALKONIUM CHLORIDE", "8001-54-5", "5974536", "Tenzaryl = benzalkonium chloride"),
    "silicate de magnesium hydrate": ("MAGNESIUM SILICATE", "1343-88-0", "23924789", ""),
    "silice epaississante":    ("SILICA", "7631-86-9", "24281", "Thickening silica"),
    "nacre hydro":             ("MICA (AND) TITANIUM DIOXIDE", "", "", "Nacreous pearlescent pigment"),
    "agent nacrant":           ("GLYCOL DISTEARATE", "627-83-8", "69988", "Pearlizing agent"),
    "agent opacifiant":        ("GLYCOL DISTEARATE", "627-83-8", "69988", "Opacifying agent; may vary by product"),
    "cire m12":                ("MICROCRYSTALLINE WAX", "63231-60-7", "24765", "M12 microcrystalline wax"),
    "cire microcristalline md": ("MICROCRYSTALLINE WAX", "63231-60-7", "24765", ""),
    "ap 7":                    ("NONOXYNOL-7", "9016-45-9", "24756940", ""),
    "non ionique ap9":         ("NONOXYNOL-9", "9016-45-9", "24756940", ""),
    "non ionique rh40":        ("PEG-40 HYDROGENATED CASTOR OIL", "61788-85-0", "24765", "Cremophor RH40"),
    "base neutralisante (tea)": ("TRIETHANOLAMINE", "102-71-6", "7618", "Base neutralisante = TEA"),
    "base neutralisante":      ("TRIETHANOLAMINE", "102-71-6", "7618", "Neutralizing base — assumed TEA"),
    "base neutraisante":       ("TRIETHANOLAMINE", "102-71-6", "7618", "OCR typo"),
    "base détetrante":         ("SURFACTANT BLEND", "", "", "Detergent base blend"),
    "silicate de sode":        ("SODIUM SILICATE", "1344-09-8", "3423151", "OCR typo of silicate de soude"),

    # ── Nonylphenol ethoxylate OCR variants ───────────────────────────────────
    "nonyl-phémol ethoxylé ap 9 m.o.e":   ("NONOXYNOL-9", "9016-45-9", "24756940", "OCR variant of Nonyl-Phenol Ethoxylé"),
    "nonyl-phéno ethoxylé ap 9 m.o.e":    ("NONOXYNOL-9", "9016-45-9", "24756940", "OCR variant"),
    "nonyl-phénoi ethoxylé ap 9 m.o.e":   ("NONOXYNOL-9", "9016-45-9", "24756940", "OCR variant"),
    "nonyl-phénoil ethoxylé ap 9 m.o.e":  ("NONOXYNOL-9", "9016-45-9", "24756940", "OCR variant"),
    "tensio-actif non ionique (ap9)":      ("NONOXYNOL-9", "9016-45-9", "24756940", "AP9 nonionic surfactant"),
    "tensioactif cationique (stepantex)":  ("DISTEARYLDIMETHYLAMMONIUM CHLORIDE", "107-64-2", "9901", "Stepantex = fabric softener quat"),

    # ── Colorants with CI numbers ─────────────────────────────────────────────
    # CI 42051: Patent Blue V — CAS 3536-49-0, PubChem CID 77073
    "colorant bleu ci 42051":    ("CI 42051", "3536-49-0", "77073", "Patent Blue V"),
    "colorant bleu patent":      ("CI 42051", "3536-49-0", "77073", "Patent Blue V"),
    "colorant bleu patente":     ("CI 42051", "3536-49-0", "77073", ""),
    "colorant rose (pink)":      ("CI 14700", "4548-53-2", "5360", "Food Red 1 / Ponceau"),
    "colorant orange red 16129": ("CI 16129", "3257-28-1", "25151780", ""),
    "colorant mauve":            ("CI 42650", "1694-09-3", "15558614", "Violet 23"),
    "colorant brun foncé à la graisse b": ("CI 26105", "6226-78-4", "24857040", "Sudan Brown"),
    "colorant vert":             ("CI 42000", "569-64-2", "11038", "Malachite green assumed"),
    "lavande : violet : (ci42051- ci42650-ci42090)": ("CI 42051 / CI 42650 / CI 42090", "", "", "Lavender shade — color blend"),
    "marine : bleu patenté (ci42051) (e131)": ("CI 42051", "3536-49-0", "77073", "Patent Blue V E131"),
    "pin-menthe : colorant vert 555d":     ("CI 42053", "4180-46-5", "6379", "Vert fixe 6B"),
    "orange ceres s":            ("CI 12140", "1229-55-6", "15552", "Orange II dye"),

    # ── Fragrance bases (all map to PARFUM per INCI convention) ───────────────
    "base parfum":                     ("PARFUM", "", "", "Fragrance base — proprietary blend"),
    "base parfum (anti nos)":          ("PARFUM", "", "", ""),
    "base parfum selon la note":       ("PARFUM", "", "", ""),
    "base parfum fougère, fraîche, sport": ("PARFUM", "", "", ""),
    "base parfum lavande, fougere, fraiche": ("PARFUM", "", "", ""),
    "base parfum lemon, navy":         ("PARFUM", "", "", ""),
    "base parfum floral/noix de coco/fruit rouge": ("PARFUM", "", "", ""),
    "base parfum fruit exotique":      ("PARFUM", "", "", ""),
    "**base parfum fruit exotique**":  ("PARFUM", "", "", ""),
    "base parfum draky":               ("PARFUM", "", "", ""),
    "base parfum elan":                ("PARFUM", "", "", ""),
    "base parfum air marin":           ("PARFUM", "", "", ""),
    "base parfum belissima":           ("PARFUM", "", "", ""),
    "base parfum creme depilatoire":   ("PARFUM", "", "", ""),
    "base parfum creme à raser":       ("PARFUM", "", "", ""),
    "base parfum crème à raser":       ("PARFUM", "", "", ""),
    "base parfum deodorant tche-hi":   ("PARFUM", "", "", ""),
    "base parfum fougere":             ("PARFUM", "", "", ""),
    "base parfum fougère":             ("PARFUM", "", "", ""),
    "base parfum fraîche":             ("PARFUM", "", "", ""),
    "base parfum lemon":               ("PARFUM", "", "", ""),
    "base parfum navy":                ("PARFUM", "", "", ""),
    "base parfum pin":                 ("PARFUM", "", "", ""),
    "base parfum pour after shave et deodorant leader": ("PARFUM", "", "", ""),
    "base parfum pour eau de cologne lavande": ("PARFUM", "", "", ""),
    "parfum (rose ; lavande ; jasmin et violette)": ("PARFUM", "", "", ""),
    "parfum hdp citron":               ("PARFUM", "", "", ""),
    "parfum jasmin f005":              ("PARFUM", "", "", ""),
    "parfum citron":                   ("PARFUM", "", "", ""),
    "parfum jasmin":                   ("PARFUM", "", "", ""),
    "parfum laque":                    ("PARFUM", "", "", ""),
    "parfum pin":                      ("PARFUM", "", "", ""),
    "parfum pour polish tableau de bord": ("PARFUM", "", "", ""),
    "parfum pour détergents":          ("PARFUM", "", "", ""),
    "parfum savon marseille":          ("PARFUM", "", "", ""),
    "huile aromatique":                ("PARFUM", "", "", "Aromatic oil — fragrance ingredient"),
    "lavande":                         ("PARFUM", "", "", "Lavender fragrance note"),
    "lemon":                           ("PARFUM", "", "", "Lemon fragrance note"),
    "navy":                            ("PARFUM", "", "", "Navy fragrance note"),
    "fougere":                         ("PARFUM", "", "", "Fougère fragrance note"),
    "fraiche":                         ("PARFUM", "", "", "Fresh fragrance note"),

    # ── Generic functional terms (unresolvable to single INCI) ───────────────
    "agent de conservation":           ("PRESERVATIVE (UNSPECIFIED)", "", "", "Generic preservative — type not specified"),
    "conservateur":                    ("PRESERVATIVE (UNSPECIFIED)", "", "", ""),
    "conservateurs":                   ("PRESERVATIVE (UNSPECIFIED)", "", "", ""),
    "agent antibactérien":             ("ANTIMICROBIAL AGENT (UNSPECIFIED)", "", "", ""),
    "agent anti corrosif":             ("CORROSION INHIBITOR (UNSPECIFIED)", "", "", ""),
    "agent anti corrosif organique":   ("CORROSION INHIBITOR (UNSPECIFIED)", "", "", ""),
    "agent anti gel":                  ("ANTIFREEZE (UNSPECIFIED)", "", "", ""),
    "solubilisant":                    ("SOLUBILIZER (UNSPECIFIED)", "", "", ""),
    "stabilisant":                     ("STABILIZER (UNSPECIFIED)", "", "", ""),
    "séquestrant":                     ("SEQUESTERING AGENT (UNSPECIFIED)", "", "", ""),
    "opacifiant":                      ("OPACIFIER (UNSPECIFIED)", "", "", ""),
    "agent apocifiant":                ("OPACIFIER (UNSPECIFIED)", "", "", "OCR typo of opacifiant"),
    "colorant":                        ("COLORANT (UNSPECIFIED)", "", "", ""),
    "colorants":                       ("COLORANT (UNSPECIFIED)", "", "", ""),
    "gaz":                             ("PROPELLANT (UNSPECIFIED)", "", "", ""),
    "gaz propulseur.":                 ("PROPELLANT (UNSPECIFIED)", "", "", ""),
    "propulseur.":                     ("PROPELLANT (UNSPECIFIED)", "", "", ""),
    "base détégrante":                 ("DETERGENT BASE (UNSPECIFIED)", "", "", "Blend — composition unknown"),
    "cemulsol":                        ("NONIONIC SURFACTANT (UNSPECIFIED)", "", "", "Cemulsol = generic nonionic surfactant trade name"),
    "pob méthyle sel sodique":         ("SODIUM METHYLPARABEN", "5026-62-0", "12270", "POB = p-hydroxybenzoate"),
    "acide sulfurique eau déminéralisée": ("SULFURIC ACID", "7664-93-9", "1118", "Sulfuric acid + water solution"),
    "cire « e »":                      ("MICROCRYSTALLINE WAX", "63231-60-7", "24765", "Cire E — shoe wax grade"),
    "cire « kle »":                    ("MICROCRYSTALLINE WAX", "63231-60-7", "24765", "Cire KLE — shoe wax grade"),
    "cire « s »":                      ("MICROCRYSTALLINE WAX", "63231-60-7", "24765", "Cire S — shoe wax grade"),

    # ── New entries from extended extraction (v2) ─────────────────────────────
    # NaOH variants
    "naoh pure":                       ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH pure"),
    "naoh à 100%":                     ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH 100%"),
    "naoh a 100%":                     ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH 100%"),
    "soude caustique pure":            ("SODIUM HYDROXIDE", "1310-73-2", "14798", "Caustic soda pure"),
    "soude caustique granulée":        ("SODIUM HYDROXIDE", "1310-73-2", "14798", "Caustic soda granules"),
    "soude caustique granule":         ("SODIUM HYDROXIDE", "1310-73-2", "14798", "Caustic soda granules"),
    "soude caustique naoh à 100 %":    ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH 100%"),
    "soude caustique naoh a 100 %":    ("SODIUM HYDROXIDE", "1310-73-2", "14798", "NaOH 100%"),
    # Butyl glycol
    "butyl glycol":                    ("BUTOXYETHANOL", "111-76-2", "8057", "Butyl glycol = Butyl cellosolve"),
    # Sodium hypochlorite
    "hypochlorite de sodium":          ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", "Bleach"),
    "hypochlorite de sodium à 48°cli": ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", "48° chlorimetric"),
    "hypochlorite de sodium à 48°chl": ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", "48° chlorimetric"),
    "hypochlorite de sodium 48°cli":   ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", ""),
    "hypochlorite de sodium 48°chl":   ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", ""),
    "hypochlorite de sodium \u00ab 48\u00b0cli \u00bb": ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", "guillemet delimiters"),
    "hypochlorite de sodium \u00ab 48\u00b0chl \u00bb": ("SODIUM HYPOCHLORITE", "7681-52-9", "23665746", "guillemet delimiters"),
    # Surfactants (new trade names)
    "tensio actif amphotaire (ninox)": ("COCAMIDOPROPYL BETAINE", "61789-40-0", "3080820", "Ninox = CAPB amphoteric"),
    "ninox":                           ("COCAMIDOPROPYL BETAINE", "61789-40-0", "3080820", "Ninox trade name"),
    "tensio actif anionique (sécosyl)":("SODIUM DODECYLBENZENESULFONATE", "25155-30-0", "23665752", "Sécosyl = LAS"),
    "sécosyl":                         ("SODIUM DODECYLBENZENESULFONATE", "25155-30-0", "23665752", "Sécosyl"),
    "secosyl":                         ("SODIUM DODECYLBENZENESULFONATE", "25155-30-0", "23665752", ""),
    # COD
    "cod":                             ("COCAMIDE DEA", "61791-31-9", "9993", "COD = Cocodiethanolamide / Cocamide DEA"),
    # Perfume bases
    "base parfum tche-hi":             ("PARFUM", "", "", "Fragrance base (trade name redacted)"),
    "base parfum nessma":              ("PARFUM", "", "", "Fragrance base (trade name redacted)"),
    "base parfum darky":               ("PARFUM", "", "", "Fragrance base (trade name redacted)"),
    "parfum lavande, rose ou autres":  ("PARFUM", "", "", "Fragrance blend"),
    "parfum lavande, marine, rose":    ("PARFUM", "", "", "Fragrance blend"),
    "parfum citron vert, lavande, rose": ("PARFUM", "", "", "Fragrance blend"),
    # Opacifier
    "agent nacron (opacifiant)":       ("GLYCOL DISTEARATE", "627-83-8", "12497", "Nacron = glycol distearate opacifier"),
    # Colorants — with PubChem CIDs resolved
    "colorant violet 1105":            ("CI 60725", "81-48-1", "6680", "Disperse Violet 1"),
    "colorant violet 11105 : parfum lavande": ("CI 60725", "81-48-1", "6680", "CI 60725 with fragrance label"),
    "colorant bleu patent : parfum marine": ("CI 42051", "3536-49-0", "77073", "Patent Blue V"),
    "colorant rose pink : parfum rose": ("CI 45410", "632-69-9", "32343", "Rose bengal / Acid Red 94"),
    "colorant jaune tartrazine : parfum citron vert": ("CI 19140", "1934-21-0", "164825", "Tartrazine"),
    "colorant rouge ponceau":          ("CI 16255", "2611-82-7", "17466", "Ponceau 4R"),
    "colorant rosole 54112 (rose fluo)": ("CI 45410", "632-69-9", "32343", "Fluorescent rose colorant"),
    # Phosphonate
    "phosphonate de sodium":           ("TETRASODIUM ETIDRONATE", "3794-83-0", "20452", "Sodium phosphonate corrosion inhibitor"),

    # ── Generic composition-only class names (from FICHE TECHNIQUE PRODUIT FINI) ─
    "cires":                           ("WAX (UNSPECIFIED)", "", "", "Generic wax class from composition list"),
    "solvant":                         ("SOLVENT (UNSPECIFIED)", "", "", "Generic solvent"),
    "emulsifiant":                     ("EMULSIFYING AGENT (UNSPECIFIED)", "", "", "Generic emulsifier"),

    # ── Fix fuzzy-matched entries missing PubChem CID ─────────────────────────
    # These were resolved via fuzzy_cosing/exact_cosing (no CID in reference),
    # so step3 CAS lookup fails (polymers/mixtures return 404 on PubChem).
    "cocamido propyl betaine":         ("COCAMIDOPROPYL BETAINE", "61789-40-0", "24769760", "Fuzzy match — explicit CID added"),
    "cocamide dea":                    ("COCAMIDE DEA", "61791-31-9", "8053426", "Exact CosIng match — explicit CID added"),

    # ── Real ingredients with CAS that previously failed PubChem lookup ───────
    "allantoine":                      ("ALLANTOIN", "97-59-6", "204", ""),
    "allantoin":                       ("ALLANTOIN", "97-59-6", "204", ""),
    "glycine":                         ("GLYCINE", "56-40-6", "750", "Amino acid humectant"),
    "saccharine sodique":              ("SODIUM SACCHARIN", "81-07-2", "5143", "Sodium saccharin sweetener"),
    "saccharine de sodium":            ("SODIUM SACCHARIN", "81-07-2", "5143", ""),
    "trichlorocarbanilide":            ("TRICLOCARBAN", "101-20-2", "7547", ""),
    "triclocarban":                    ("TRICLOCARBAN", "101-20-2", "7547", ""),
    "oxyquinoléine sulfate":           ("OXYQUINOLINE SULFATE", "134-31-6", "517000", "Antiseptic/preservative"),
    "oxyquinoléine de sulfate":        ("OXYQUINOLINE SULFATE", "134-31-6", "517000", ""),
    "thioglycolate de calcium":        ("CALCIUM THIOGLYCOLATE", "814-71-1", "13141", "Depilatory active"),
}


def nfc(s: str) -> str:
    """Unicode NFC normalize + lowercase + strip."""
    return unicodedata.normalize("NFC", s).lower().strip()


def strip_concentration(name: str) -> str:
    """Remove concentration suffixes like '70%', '50%', 'à 96°' etc."""
    name = re.sub(r"\s*[\(\[]?[0-9]+[.,]?[0-9]*\s*%[^\)]*[\)\]]?", "", name)
    name = re.sub(r"\s+à\s+[0-9]+°", "", name)
    name = re.sub(r"\s*\(v/v\)", "", name)
    return name.strip()


def load_cosing(path: Path) -> pd.DataFrame:
    """Load CosIng CSV (has 6 metadata rows before the actual header)."""
    df = pd.read_csv(path, skiprows=6, encoding="utf-8", on_bad_lines="skip",
                     dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["inci_lower"] = df["INCI name"].fillna("").apply(nfc)
    return df


def load_beauteeru(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["inci_lower"] = df["name"].fillna("").apply(nfc)
    return df


def exact_match(
    raw_lower: str,
    cosing: pd.DataFrame,
    beauteeru: pd.DataFrame,
) -> dict | None:
    """Try exact (NFC-normalised, lowercase) match in both reference sets."""
    # CosIng first
    hits = cosing[cosing["inci_lower"] == raw_lower]
    if not hits.empty:
        row = hits.iloc[0]
        return {
            "inci_name": str(row["INCI name"]).strip(),
            "cas_number": str(row.get("CAS No", "") or "").strip(),
            "pubchem_cid": "",
            "match_method": "exact_cosing",
            "match_score": 100,
            "confidence": "high",
            "notes": "",
        }
    # beauteeru
    hits = beauteeru[beauteeru["inci_lower"] == raw_lower]
    if not hits.empty:
        row = hits.iloc[0]
        return {
            "inci_name": str(row["name"]).strip().upper(),
            "cas_number": str(row.get("casNo", "") or "").strip(),
            "pubchem_cid": str(row.get("pubchem_cid", "") or "").strip(),
            "match_method": "exact_beauteeru",
            "match_score": 100,
            "confidence": "high",
            "notes": "",
        }
    return None


def fuzzy_match(
    raw_lower: str,
    cosing_names: list[str],
    cosing: pd.DataFrame,
    threshold: int = FUZZY_THRESHOLD,
) -> dict | None:
    """token_sort_ratio fuzzy match against CosIng INCI names."""
    if FUZZY_LIB is None:
        return None
    if FUZZY_LIB == "rapidfuzz":
        result = rfprocess.extractOne(
            raw_lower, cosing_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            best_name, score, _ = result
        else:
            return None
    else:
        result = fuzz.process.extractOne(
            raw_lower, cosing_names,
            scorer=fuzz.token_sort_ratio,
        )
        if result is None or result[1] < threshold:
            return None
        best_name, score = result[0], result[1]

    hits = cosing[cosing["inci_lower"] == best_name]
    if hits.empty:
        return None
    row = hits.iloc[0]
    conf = "high" if score >= 90 else "medium" if score >= 82 else "low"
    return {
        "inci_name": str(row["INCI name"]).strip(),
        "cas_number": str(row.get("CAS No", "") or "").strip(),
        "pubchem_cid": "",
        "match_method": "fuzzy_cosing",
        "match_score": round(score, 1),
        "confidence": conf,
        "notes": f"fuzzy matched '{raw_lower}' -> '{row['INCI name'].strip()}'",
    }


def llm_match(names: list[str], client: "anthropic.Anthropic") -> dict[str, dict]:
    """
    Batch-call Claude API for unresolved French ingredient names.
    Returns dict: raw_name_lower -> result_dict
    """
    if not names:
        return {}

    # Build a numbered list for the prompt
    numbered = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
    prompt = f"""You are an expert cosmetic chemist and INCI nomenclature specialist.

Below is a list of cosmetic ingredient names from French industrial documentation.
For each name, provide:
- The standard INCI name (in uppercase English)
- The CAS registry number (or "N/A" if unknown)
- Confidence: HIGH / MEDIUM / LOW
- A brief note explaining the mapping (trade name, abbreviation, etc.)

If a name is a mixture, fragrance base, or generic functional term that cannot be resolved to a single INCI compound, use INCI = "UNRESOLVABLE" and explain why.

Respond ONLY with a JSON array, one object per input, with keys:
  "index" (1-based), "inci_name", "cas_number", "confidence", "notes"

Ingredient names to resolve:
{numbered}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extract JSON array from response
    import json
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if not json_match:
        print(f"  WARNING: Claude response did not contain JSON array. Raw:\n{text[:300]}")
        return {}

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}. Raw:\n{text[:300]}")
        return {}

    results: dict[str, dict] = {}
    for item in items:
        idx = int(item.get("index", 0)) - 1
        if idx < 0 or idx >= len(names):
            continue
        raw = names[idx]
        inci = item.get("inci_name", "").strip().upper()
        cas  = item.get("cas_number", "").strip()
        conf = item.get("confidence", "LOW").upper()
        notes = item.get("notes", "").strip()

        if inci == "UNRESOLVABLE" or not inci:
            results[raw] = {
                "inci_name": "",
                "cas_number": "",
                "pubchem_cid": "",
                "match_method": "llm_unresolvable",
                "match_score": 0,
                "confidence": "low",
                "notes": notes,
            }
        else:
            results[raw] = {
                "inci_name": inci,
                "cas_number": cas if cas != "N/A" else "",
                "pubchem_cid": "",
                "match_method": "llm_claude",
                "match_score": 0,
                "confidence": conf.lower(),
                "notes": notes,
            }
    return results


def main(skip_llm: bool = False) -> None:
    # ── Load inputs ────────────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run step1 first.", file=sys.stderr)
        sys.exit(1)

    raw_df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    raw_names = raw_df["raw_name"].tolist()
    print(f"Loaded {len(raw_names)} raw ingredient names from step 1")

    print(f"Loading CosIng ({COSING_CSV.name})...")
    cosing = load_cosing(COSING_CSV)
    print(f"  {len(cosing):,} INCI entries")

    print(f"Loading beauteeru crosswalk ({BEAUTEERU_CSV.name})...")
    beauteeru = load_beauteeru(BEAUTEERU_CSV)
    print(f"  {len(beauteeru):,} entries")

    cosing_names_lower = cosing["inci_lower"].tolist()

    # ── Optionally init Claude client ──────────────────────────────────────────
    claude_client = None
    if not skip_llm:
        if not HAS_ANTHROPIC:
            print("WARNING: anthropic package not installed. Skipping LLM stage.")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY") or _load_env_key()
            if api_key:
                import anthropic
                claude_client = anthropic.Anthropic(api_key=api_key)
                print("Claude API client ready.")
            else:
                print("WARNING: ANTHROPIC_API_KEY not found. Skipping LLM stage.")

    # ── Process each name ──────────────────────────────────────────────────────
    results: list[dict] = []
    llm_queue: list[str] = []   # raw_lower names deferred to LLM

    counts = {"manual": 0, "exact": 0, "fuzzy": 0, "llm": 0,
              "unresolvable": 0, "skipped": 0, "pending_llm": 0}

    for raw in raw_names:
        raw_stripped = strip_concentration(raw)
        raw_lower    = nfc(raw_stripped)
        raw_lower_orig = nfc(raw)   # also try with concentration suffix

        # Skip non-ingredients (water, solvents, metadata noise)
        if raw_lower in SKIP_NAMES or raw_lower_orig in SKIP_NAMES:
            results.append({
                "raw_name": raw, "inci_name": "WATER/SOLVENT",
                "cas_number": "", "pubchem_cid": "",
                "match_method": "skipped_solvent",
                "match_score": 100, "confidence": "high", "notes": "water/solvent",
            })
            counts["skipped"] += 1
            continue

        # Skip document fragments: very long strings or HTML content
        _is_doc_fragment = (
            len(raw) > 200
            or "<table>" in raw.lower()
            or "# specifications" in raw.lower()
            or "# conditionnement" in raw.lower()
            or raw.strip().startswith("consulter un médecin")
            or raw.strip().startswith("marron et incolore")
            or re.search(r"fabrication de |portée des enfants|évoluer\.", raw)
        )
        if _is_doc_fragment:
            results.append({
                "raw_name": raw, "inci_name": "",
                "cas_number": "", "pubchem_cid": "",
                "match_method": "skipped_noise",
                "match_score": 0, "confidence": "low",
                "notes": "Document fragment / metadata noise — not an ingredient",
            })
            counts["skipped"] += 1
            continue

        # Stage 0: Manual map (abbreviations + known trade names)
        if raw_lower in MANUAL_MAP or raw_lower_orig in MANUAL_MAP:
            key = raw_lower if raw_lower in MANUAL_MAP else raw_lower_orig
            inci, cas, cid, notes = MANUAL_MAP[key]
            results.append({
                "raw_name": raw, "inci_name": inci,
                "cas_number": cas, "pubchem_cid": cid,
                "match_method": "manual_map",
                "match_score": 100, "confidence": "high", "notes": notes,
            })
            counts["manual"] += 1
            continue

        # Stage A: Exact match
        hit = exact_match(raw_lower, cosing, beauteeru) or \
              exact_match(raw_lower_orig, cosing, beauteeru)
        if hit:
            hit["raw_name"] = raw
            results.append(hit)
            counts["exact"] += 1
            continue

        # Stage B: Fuzzy match
        hit = fuzzy_match(raw_lower, cosing_names_lower, cosing) or \
              fuzzy_match(raw_lower_orig, cosing_names_lower, cosing)
        if hit:
            hit["raw_name"] = raw
            results.append(hit)
            counts["fuzzy"] += 1
            continue

        # Stage C: Queue for LLM
        llm_queue.append(raw)
        results.append({
            "raw_name": raw, "inci_name": "",
            "cas_number": "", "pubchem_cid": "",
            "match_method": "pending_llm",
            "match_score": 0, "confidence": "", "notes": "",
        })
        counts["pending_llm"] += 1

    # ── Stage C: LLM batch ─────────────────────────────────────────────────────
    if llm_queue and claude_client:
        print(f"\nStage C: Sending {len(llm_queue)} unresolved names to Claude API...")
        # Process in batches of 30 to stay within prompt limits
        BATCH = 30
        llm_results: dict[str, dict] = {}
        for i in range(0, len(llm_queue), BATCH):
            batch = llm_queue[i:i + BATCH]
            batch_lower = [nfc(strip_concentration(n)) for n in batch]
            print(f"  Batch {i//BATCH + 1}/{(len(llm_queue)-1)//BATCH + 1}: {len(batch)} names")
            batch_results = llm_match(batch_lower, claude_client)
            # Map back using index position
            for j, orig_name in enumerate(batch):
                key = nfc(strip_concentration(orig_name))
                if key in batch_results:
                    llm_results[orig_name] = batch_results[key]
            time.sleep(0.5)  # gentle rate limiting

        # Update results
        for i, row in enumerate(results):
            if row["match_method"] == "pending_llm":
                raw = row["raw_name"]
                if raw in llm_results:
                    lr = llm_results[raw]
                    results[i].update(lr)
                    results[i]["raw_name"] = raw
                    if lr["match_method"] == "llm_unresolvable":
                        counts["unresolvable"] += 1
                    else:
                        counts["llm"] += 1
                    counts["pending_llm"] -= 1

    elif llm_queue and not claude_client:
        print(f"\n{len(llm_queue)} names remain unresolved (LLM skipped).")
        print("Re-run without --skip-llm (and with ANTHROPIC_API_KEY set) to resolve them.")

    # ── Write output ───────────────────────────────────────────────────────────
    fields = ["raw_name", "inci_name", "cas_number", "pubchem_cid",
              "match_method", "match_score", "confidence", "notes"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"\nWrote {len(results)} rows -> {OUTPUT_CSV}")
    print("\n=== SUMMARY ===")
    total = len(results)
    resolved = counts["manual"] + counts["exact"] + counts["fuzzy"] + counts["llm"]
    print(f"  Total ingredients processed : {total}")
    print(f"  Stage 0 manual map          : {counts['manual']}")
    print(f"  Stage A exact match         : {counts['exact']}")
    print(f"  Stage B fuzzy match         : {counts['fuzzy']}")
    print(f"  Stage C LLM resolved        : {counts['llm']}")
    print(f"  LLM unresolvable            : {counts['unresolvable']}")
    print(f"  Solvents/skipped            : {counts['skipped']}")
    print(f"  Still pending (no LLM run)  : {counts['pending_llm']}")
    print(f"  INCI coverage (excl. water) : {resolved}/{total - counts['skipped']} = "
          f"{100*resolved/max(1, total - counts['skipped']):.1f}%")


def _load_env_key() -> str | None:
    """Try to load ANTHROPIC_API_KEY from rapport3/.env"""
    env_path = PROJECT_ROOT / "rapport3" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Claude API stage (run stages 0+A+B only)")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm)
