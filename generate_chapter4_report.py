"""
generate_chapter4_report.py — run from your project root: python generate_chapter4_report.py

Reads every artifact your pipeline already produces and prints READY-TO-PASTE
markdown for every table in Chapter Four. No manual transcription needed --
copy each printed block directly into your dissertation.

If a required file is missing, this tells you exactly which command to run
to produce it, rather than failing silently.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import METRICS_DIR, PROCESSED_DATA_DIR


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def require(path, command_to_fix):
    if not path.exists():
        print(f"MISSING: {path}")
        print(f"  -> Run this first: {command_to_fix}")
        return False
    return True


# ---------------------------------------------------------------------------
section("TABLE 4.1 — Dataset cleaning summary")
# ---------------------------------------------------------------------------
prep_report_path = METRICS_DIR / "preprocessing_report.json"
if not prep_report_path.exists():
    print("No saved preprocessing report found. Run this once to generate it")
    print("(safe to run even if you've already got train.csv/test.csv -- it")
    print("re-runs preprocessing and saves the report without changing your")
    print("already-cleaned, already-domain-corrected files if you skip re-saving):\n")
    print("  python3 -c \"")
    print("from src.preprocessing import run_preprocessing_pipeline")
    print("import json")
    print("_, _, report = run_preprocessing_pipeline(save_outputs=False)")
    print("with open('results/metrics/preprocessing_report.json', 'w') as f:")
    print("    json.dump(report.to_dict(), f, indent=2)")
    print("print(report.to_dict())\"")
else:
    with open(prep_report_path) as f:
        pr = json.load(f)
    domain_clean_note = (
        "4,142 (3,212 IETF + 930 across ~27 other domains)"
        if True else ""
    )
    print("| Stage | Row count | Note |")
    print("|---|---|---|")
    print(f"| Raw dataset (post label-mapping fix) | {pr['raw_row_count']:,} | — |")
    print(f"| Duplicate URLs removed | {pr['duplicate_rows_removed']:,} | Removed before splitting |")
    print(f"| Rows with missing/empty URL removed | {pr['missing_value_rows_removed']:,} | — |")
    print(f"| Rows with unrecognized label removed | {pr['unmapped_label_rows_removed']:,} | — |")
    print(f"| Rows corrected via domain-based label cleaning | *(see your own verify_dataset_labels.py output)* | Run separately, not part of this report |")
    print(f"| Final class balance (Safe / Malicious) | {pr['class_balance']} | — |")
    print(f"| Train rows | {pr['train_row_count']:,} |")
    print(f"| Test rows | {pr['test_row_count']:,} |")

# ---------------------------------------------------------------------------
section("TABLE 4.2 — Model comparison")
# ---------------------------------------------------------------------------
mc_path = METRICS_DIR / "model_comparison.csv"
if require(mc_path, "python -m src.train"):
    import pandas as pd
    df = pd.read_csv(mc_path, index_col=0)
    print("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training time (s) |")
    print("|---|---|---|---|---|---|---|")
    for model_name, row in df.iterrows():
        print(f"| {model_name} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
              f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
              f"{row['training_time_seconds']:.2f} |")

with open(METRICS_DIR / "best_model_metadata.json") if (METRICS_DIR / "best_model_metadata.json").exists() else open(Path("models/best_model_metadata.json")) as f:
    meta = json.load(f)
print(f"\nSelected model (for your prose): {meta['model_name']}")

# ---------------------------------------------------------------------------
section("SECTION 4.3 — Classification report + cross-validation")
# ---------------------------------------------------------------------------
cr_path = METRICS_DIR / "classification_report.txt"
if require(cr_path, "python -m src.evaluate"):
    print(cr_path.read_text())

cv_path = METRICS_DIR / "cross_validation_scores.json"
if require(cv_path, "python -m src.evaluate"):
    with open(cv_path) as f:
        cv = json.load(f)
    print("Cross-validation summary sentence to paste into your prose:")
    parts = [f"{k}: {v['mean']:.4f} (SD {v['std']:.4f})" for k, v in cv.items()]
    print("  " + "; ".join(parts))

# ---------------------------------------------------------------------------
section("TABLE 4.3 — Full SHAP feature importance ranking")
# ---------------------------------------------------------------------------
shap_path = METRICS_DIR / "shap_top_features.json"
if require(shap_path, "python -m src.explainability"):
    with open(shap_path) as f:
        shap_ranking = json.load(f)
    print("| Rank | Feature | Mean |SHAP value| |")
    print("|---|---|---|")
    for i, item in enumerate(shap_ranking, 1):
        print(f"| {i} | {item['feature']} | {item['mean_abs_shap']:.4f} |")

# ---------------------------------------------------------------------------
section("SECTION 4.6/4.7 — LIME-SHAP comparison and stability")
# ---------------------------------------------------------------------------
ls_path = METRICS_DIR / "lime_shap_comparison.json"
if require(ls_path, "python lime_shap_analysis.py"):
    with open(ls_path) as f:
        ls = json.load(f)
    print("Sentences to paste into your prose:")
    print(f"  Mean agreement: {ls['mean_agreement_jaccard']:.3f} "
          f"(std = {ls['std_agreement_jaccard']:.3f})")
    print(f"  Mean LIME time per explanation: {ls['mean_lime_time_seconds']:.3f}s")
    print(f"  Mean SHAP time per explanation: {ls['mean_shap_time_seconds']:.3f}s")
    print(f"  LIME stability: {ls['lime_stability']:.3f}")
    print(f"  SHAP stability: {ls['shap_stability']:.3f}")
    print(f"  Per-instance agreement values: {[round(a, 2) for a in ls['per_instance_agreement']]}")

print("\n" + "=" * 70)
print("DONE. Copy each block above directly into the corresponding table/")
print("sentence in Chapter Four. If any section printed a MISSING file,")
print("run the suggested command, then re-run this script.")
print("=" * 70)