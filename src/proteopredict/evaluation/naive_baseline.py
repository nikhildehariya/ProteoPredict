"""
ProteoPredict - CAFA Naive Frequency Baseline & Information Accretion (IA) Engine
Computes standard CAFA naive baseline (marginal prior class probabilities)
and Information Accretion weights IA(c) = -log2(P(c)) to quantify term specificity.
"""

import json
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.NaiveBaseline")


def compute_naive_and_ia(data_dir: str = "data/processed", results_dir: str = "results"):
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    train_data = np.load(data_path / "train_data.npz")
    val_data = np.load(data_path / "val_data.npz")
    test_data = np.load(data_path / "test_data.npz")

    y_train = train_data["y"]
    y_val = val_data["y"]
    y_test = test_data["y"]

    with open(data_path / "go_info.json", "r", encoding="utf-8") as f:
        go_info = json.load(f)
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})
    num_classes = len(classes)

    # 1. Compute Marginal Class Prior P(c) on Train Set
    train_counts = y_train.sum(axis=0)
    total_train = len(y_train)
    p_c = train_counts / total_train

    # 2. Information Accretion (Shannon Information Content)
    # IA(c) = -log2(P(c))
    eps = 1e-7
    ia_weights = -np.log2(np.clip(p_c, eps, 1.0))
    ia_dict = {
        classes[i]: {
            "p_train": float(p_c[i]),
            "ia_weight": float(ia_weights[i]),
            "name": go_metadata.get(classes[i], {}).get("name", "Unknown"),
            "category": go_metadata.get(classes[i], {}).get("category", "Unknown")
        }
        for i in range(num_classes)
    }

    with open(data_path / "ia_weights.json", "w", encoding="utf-8") as f:
        json.dump(ia_dict, f, indent=2)
    logger.info(f"Saved Information Accretion weights for {num_classes} terms to {data_path / 'ia_weights.json'}")

    # 3. Naive Baseline Prediction
    # For every protein in val and test, predict p_c for all classes
    val_probs = np.tile(p_c, (len(y_val), 1))
    test_probs = np.tile(p_c, (len(y_test), 1))

    # Calibrate optimal decision threshold tau* on validation set
    thresholds = np.linspace(0.05, 0.70, 60)
    best_f1 = 0.0
    best_tau = 0.20

    for t in thresholds:
        y_val_bin = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, y_val_bin, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(t)

    logger.info(f"Naive baseline calibrated threshold on validation: tau* = {best_tau:.4f} (Val F1: {best_f1:.4f})")

    # Evaluate on test set using tau*
    y_test_bin = (test_probs >= best_tau).astype(int)
    prec_micro = float(precision_score(y_test, y_test_bin, average="micro", zero_division=0))
    rec_micro = float(recall_score(y_test, y_test_bin, average="micro", zero_division=0))
    f1_micro = float(f1_score(y_test, y_test_bin, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_test_bin, average="macro", zero_division=0))

    mf_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "MF"]
    bp_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "BP"]
    cc_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "CC"]

    f1_mf = float(f1_score(y_test[:, mf_indices], y_test_bin[:, mf_indices], average="micro", zero_division=0)) if mf_indices else 0.0
    f1_bp = float(f1_score(y_test[:, bp_indices], y_test_bin[:, bp_indices], average="micro", zero_division=0)) if bp_indices else 0.0
    f1_cc = float(f1_score(y_test[:, cc_indices], y_test_bin[:, cc_indices], average="micro", zero_division=0)) if cc_indices else 0.0

    naive_metrics = {
        "model": "Naive Marginal Frequency Baseline (CAFA Standard)",
        "calibrated_threshold": best_tau,
        "validation_fmax": best_f1,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_mf": f1_mf,
        "f1_bp": f1_bp,
        "f1_cc": f1_cc
    }

    with open(results_path / "naive_baseline.json", "w", encoding="utf-8") as f:
        json.dump(naive_metrics, f, indent=2)

    logger.info("=" * 50)
    logger.info("NAIVE BASELINE RESULTS (CAFA Reference)")
    logger.info("=" * 50)
    logger.info(f"Calibrated Threshold tau*: {best_tau:.3f}")
    logger.info(f"Test Micro-F1 (F_max):    {f1_micro:.4f}")
    logger.info(f"Test Macro-F1:            {f1_macro:.4f}")
    logger.info(f"F1 (Molecular Function):  {f1_mf:.4f}")
    logger.info(f"F1 (Biological Process):  {f1_bp:.4f}")
    logger.info(f"F1 (Cellular Component):  {f1_cc:.4f}")
    logger.info("=" * 50)

    return naive_metrics, ia_dict


if __name__ == "__main__":
    compute_naive_and_ia()
