"""
ProteoPredict - Standard Bioinformatics Evaluation Protocol
Evaluates models using validation-calibrated optimal threshold (zero test leakage)
and computes CAFA benchmark metrics (F_max, Micro/Macro F1, Precision, Recall, Per-Category F1).
"""

import json
import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.Evaluate")


def evaluate_model_on_test(
    model_path: str = "models/best_model.h5",
    data_dir: str = "data/processed",
    threshold: float = None,
    results_dir: str = "results"
):
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Preprocessed Data
    test_npz_path = data_dir / "test_data.npz"
    val_npz_path = data_dir / "val_data.npz"

    if not test_npz_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_npz_path}")

    logger.info(f"Loading test dataset from: {test_npz_path}")
    test_data = np.load(test_npz_path)
    X_test, y_test = test_data["X"], test_data["y"]

    with open(data_dir / "go_info.json", "r", encoding="utf-8") as f:
        go_info = json.load(f)
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})

    # 2. Determine Optimal Threshold Strictly on Validation Set (Zero Test Leakage)
    if threshold is None:
        metrics_file = model_path.parent / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                saved_metrics = json.load(f)
            threshold = float(saved_metrics.get("calibrated_threshold", 0.20))
            logger.info(f"Loaded validation-calibrated threshold from {metrics_file}: tau* = {threshold:.4f}")
        else:
            logger.info("metrics.json not found; calibrating optimal threshold on validation set...")
            val_data = np.load(val_npz_path)
            X_val, y_val = val_data["X"], val_data["y"]

            # Load model and predict on validation
            model = load_model(model_path, compile=False)
            val_probs = model.predict(X_val, batch_size=64, verbose=0)

            best_f1 = 0.0
            best_t = 0.20
            for t in np.linspace(0.02, 0.60, 50):
                f1 = f1_score(y_val, (val_probs >= t).astype(int), average="micro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            threshold = best_t
            logger.info(f"Calibrated threshold tau* on validation set: {threshold:.4f} (Val F1: {best_f1:.4f})")

    # 3. Load Model and Predict on Test Set
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    logger.info("Generating predictions on test set (unseen, zero homology leakage)...")
    y_pred_proba = model.predict(X_test, batch_size=64, verbose=1)

    # 4. Evaluate Strictly Using the Validation-Calibrated Threshold
    y_pred_bin = (y_pred_proba >= threshold).astype(int)

    precision_micro = float(precision_score(y_test, y_pred_bin, average="micro", zero_division=0))
    recall_micro = float(recall_score(y_test, y_pred_bin, average="micro", zero_division=0))
    f1_micro = float(f1_score(y_test, y_pred_bin, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred_bin, average="macro", zero_division=0))

    # Per-category metrics
    mf_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "MF"]
    bp_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "BP"]
    cc_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "CC"]

    f1_mf = float(f1_score(y_test[:, mf_indices], y_pred_bin[:, mf_indices], average="micro", zero_division=0)) if mf_indices else 0.0
    f1_bp = float(f1_score(y_test[:, bp_indices], y_pred_bin[:, bp_indices], average="micro", zero_division=0)) if bp_indices else 0.0
    f1_cc = float(f1_score(y_test[:, cc_indices], y_pred_bin[:, cc_indices], average="micro", zero_division=0)) if cc_indices else 0.0

    eval_results = {
        "model_path": str(model_path),
        "test_samples": int(len(X_test)),
        "num_classes": int(len(classes)),
        "validation_calibrated_threshold": threshold,
        "test_precision_micro": precision_micro,
        "test_recall_micro": recall_micro,
        "test_f1_micro": f1_micro,
        "test_f1_macro": f1_macro,
        "test_f1_molecular_function": f1_mf,
        "test_f1_biological_process": f1_bp,
        "test_f1_cellular_component": f1_cc,
    }

    print("\n" + "=" * 60)
    print("STANDARD BIOINFORMATICS EVALUATION REPORT (TEST SET)")
    print("=" * 60)
    print(f"Test Samples:                    {len(X_test):,}")
    print(f"GO Terms Evaluated:              {len(classes)} (MF, BP, CC)")
    print(f"Decision Threshold:              {threshold:.4f} (Validation-Calibrated)")
    print("-" * 60)
    print(f"Precision (Micro):               {precision_micro:.4f}")
    print(f"Recall (Micro):                  {recall_micro:.4f}")
    print(f"F1-Score (Micro / CAFA F_max):   {f1_micro:.4f}")
    print(f"F1-Score (Macro):                {f1_macro:.4f}")
    print("-" * 60)
    print("Per-Ontology Category Performance:")
    print(f"  Molecular Function (MF) F1:    {f1_mf:.4f}")
    print(f"  Biological Process (BP) F1:    {f1_bp:.4f}")
    print(f"  Cellular Component (CC) F1:    {f1_cc:.4f}")
    print("=" * 60 + "\n")

    eval_output_file = results_dir / "test_evaluation_report.json"
    with open(eval_output_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Saved evaluation report to {eval_output_file}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProteoPredict model on test data.")
    parser.add_argument("--model_path", type=str, default="models/best_model.h5", help="Path to trained model.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: loaded from metrics.json).")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save evaluation results.")
    args = parser.parse_args()

    evaluate_model_on_test(args.model_path, args.data_dir, args.threshold, args.results_dir)


if __name__ == "__main__":
    main()