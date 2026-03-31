# Baseline NER Comparison

This folder documents the comparison between the INCI pipeline and existing English-language
chemical named-entity recognition (NER) tools when applied to French industrial ingredient text.

## Summary

Three established systems — **ChemDataExtractor**, **ChemSpot**, and **OSCAR** — were evaluated
on a sample of 10 OCR-converted Markdown documents from the ENAD industrial corpus (47 unique
ingredient mentions). All three systems returned zero correct ingredient identifications.

This result is expected: all three systems were trained exclusively on English scientific literature
and are not designed to handle French industrial trade names, abbreviated forms, or grammatical
inversion patterns. It empirically validates the need for the dedicated pipeline described in the paper.

## Experimental setup

| Parameter | Value |
|-----------|-------|
| Documents tested | 10 OCR-converted Markdown files |
| Unique ingredient mentions | 47 |
| ChemDataExtractor version | 2.3.0 |
| ChemSpot version | 2.0 |
| OSCAR version | 4.0 |
| Input format | Plain-text Markdown (UTF-8) |
| Entity types targeted (CDE) | Chemical, CompoundHeading |

## Results

| System | Correct identifications | False positives | Notes |
|--------|------------------------|-----------------|-------|
| ChemDataExtractor | 0 / 47 | 3 | Partial English header fragments misidentified |
| ChemSpot | 0 / 47 | 0 | No entities extracted at all |
| OSCAR | 0 / 47 | 1 | Single false match on "Na" fragment |

## Representative failure examples

The following ingredient strings from the test documents were unrecognized by all three systems:

```
"Lauryl-Ether Sulfate de Sodium LES-Na 70%"  → expected: SODIUM LAURETH SULFATE
"Texapon N70"                                 → expected: SODIUM LAURETH SULFATE
"acide gras de coco"                          → expected: COCONUT ACID
"Kathon CG"                                   → expected: METHYLCHLOROISOTHIAZOLINONE (AND) METHYLISOTHIAZOLINONE
"AP9"                                         → expected: NONOXYNOL-9
"Comperlan KD"                                → expected: COCAMIDE DEA
"DDBSH"                                       → expected: SODIUM DODECYLBENZENESULFONATE
"BHA"                                         → expected: BHA (Butylhydroxyanisole)
"MPG"                                         → expected: PROPYLENE GLYCOL
"silicate de soude"                           → expected: SODIUM SILICATE
```

## Why these systems fail on French industrial text

1. **Language mismatch**: CDE and ChemSpot tokenize on whitespace and apply English morphological
   rules; French particles ("de", "d'", "du") and word-order inversion prevent chemical name
   segmentation.

2. **Trade name vocabulary gap**: Industrial trade names ("Texapon N70", "Kathon CG", "AP9",
   "DDBSH") do not appear in the PubChem, ChEMBL, or ChemNER training corpora used by these tools.

3. **Abbreviation handling**: Single-character or short-code abbreviations (LES, BHA, MPG, AP9)
   are not in chemical dictionaries and are not parseable without domain context.

## Reproduce this experiment

The test documents used in this experiment are synthetic anonymized equivalents of the original
ENAD corpus documents (no proprietary product names or formulas). They are provided in
`test_documents/` as plain-text Markdown files.

```bash
# Install dependencies
pip install chemdataextractor2

# Run CDE on test documents
python run_cde_baseline.py

# Results written to cde_results.json
```

> **Note:** ChemSpot requires Java 8+ and can be run from its official distribution.
> OSCAR requires the OSCAR4 API jar. Detailed setup instructions for both are in
> `setup_chemspot.md` and `setup_oscar.md`.

## Reference

This experiment is described in Section 5.2 of:

> Aroua A.H., Boukandoura M., Brakta K., Meziari R. (2026).
> *Automated Digitalization and Molecular Enrichment of Industrial Chemical Documentation:
> A French-Language NLP and Cheminformatics Pipeline.* [Journal TBD]
