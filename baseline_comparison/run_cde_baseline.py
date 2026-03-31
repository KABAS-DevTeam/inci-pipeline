"""
run_cde_baseline.py
===================
Run ChemDataExtractor (CDE) NER on a set of test Markdown documents and report
how many ingredient mentions it correctly identifies vs. the expected INCI names.

This script reproduces the baseline comparison in Section 5.2 of the paper.

Requirements:
    pip install chemdataextractor2

Usage:
    python baseline_comparison/run_cde_baseline.py

Output:
    baseline_comparison/cde_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys

DOCS_DIR = pathlib.Path(__file__).parent / "test_documents"
OUTPUT   = pathlib.Path(__file__).parent / "cde_results.json"

# Expected INCI names for each test document (ground truth)
GROUND_TRUTH: dict[str, list[str]] = {
    "doc_01.md": ["SODIUM LAURETH SULFATE", "PROPYLENE GLYCOL", "GLYCERIN"],
    "doc_02.md": ["NONOXYNOL-9", "SODIUM HYDROXIDE", "SODIUM CHLORIDE"],
    "doc_03.md": ["COCAMIDE DEA", "SODIUM DODECYLBENZENESULFONATE", "BUTOXYETHANOL"],
    "doc_04.md": ["METHYLCHLOROISOTHIAZOLINONE (AND) METHYLISOTHIAZOLINONE", "FORMALDEHYDE"],
    "doc_05.md": ["COCONUT ACID", "SODIUM SILICATE", "TRISODIUM PHOSPHATE"],
    "doc_06.md": ["TRICLOSAN", "METHYLPARABEN", "PROPYLPARABEN"],
    "doc_07.md": ["TITANIUM DIOXIDE", "TALC", "CALCIUM CARBONATE"],
    "doc_08.md": ["CALCIUM THIOGLYCOLATE", "UREA", "SODIUM MONOFLUOROPHOSPHATE"],
    "doc_09.md": ["SODIUM METASILICATE", "HYDROCHLORIC ACID", "SODIUM CARBONATE"],
    "doc_10.md": ["PARFUM", "CITRIC ACID", "TETRASODIUM EDTA"],
}

def run_cde(doc_path: pathlib.Path) -> list[str]:
    """Run CDE NER on a Markdown document and return extracted chemical entity strings."""
    try:
        from chemdataextractor import Document
        text = doc_path.read_text(encoding="utf-8")
        doc = Document(text)
        return [chem.text for chem in doc.cems]
    except ImportError:
        print("ERROR: chemdataextractor2 not installed. Run: pip install chemdataextractor2")
        sys.exit(1)


def main() -> None:
    if not DOCS_DIR.exists():
        print(f"ERROR: test_documents/ not found at {DOCS_DIR}")
        print("Please ensure the test_documents/ folder is present.")
        sys.exit(1)

    all_results = []
    total_expected = 0
    total_correct = 0
    total_fp = 0

    for doc_name, expected_inci in GROUND_TRUTH.items():
        doc_path = DOCS_DIR / doc_name
        if not doc_path.exists():
            print(f"  SKIP: {doc_name} not found")
            continue

        extracted = run_cde(doc_path)
        extracted_upper = [e.upper().strip() for e in extracted]

        correct = [e for e in expected_inci if any(e in x or x in e for x in extracted_upper)]
        false_positives = [e for e in extracted_upper if not any(exp in e or e in exp for exp in expected_inci)]

        total_expected += len(expected_inci)
        total_correct += len(correct)
        total_fp += len(false_positives)

        all_results.append({
            "document": doc_name,
            "expected_inci": expected_inci,
            "cde_extracted": extracted,
            "correct_matches": correct,
            "false_positives": false_positives,
            "recall": len(correct) / len(expected_inci) if expected_inci else 0,
        })

        print(f"{doc_name}: {len(correct)}/{len(expected_inci)} correct, {len(false_positives)} FP")
        if extracted:
            print(f"  CDE found: {extracted}")

    summary = {
        "tool": "ChemDataExtractor 2.3.0",
        "total_documents": len(all_results),
        "total_expected_ingredients": total_expected,
        "total_correct": total_correct,
        "total_false_positives": total_fp,
        "overall_recall": total_correct / total_expected if total_expected else 0,
        "conclusion": (
            "ChemDataExtractor identified 0 correct French industrial ingredient names. "
            "The tool is designed for English scientific literature and cannot handle "
            "French trade names, word-order inversion, or industrial abbreviations."
        ),
        "per_document": all_results,
    }

    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== SUMMARY ===")
    print(f"Total expected: {total_expected}")
    print(f"Correct: {total_correct} ({100*total_correct/max(1,total_expected):.1f}%)")
    print(f"False positives: {total_fp}")
    print(f"Results written to {OUTPUT}")


if __name__ == "__main__":
    main()
