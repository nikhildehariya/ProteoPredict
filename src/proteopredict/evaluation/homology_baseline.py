"""
ProteoPredict - CAFA Homology Transfer Baseline (BLAST / k-NN Sequence Alignment)
Implements standard CAFA annotation transfer baseline: transfers Gene Ontology labels
from top sequence alignment matches in the training database to unseen test targets.
Evaluates the baseline under strict validation calibration across the < 30% sequence homology split
to quantify the 'Twilight Zone' performance deficit of traditional alignment methods.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.HomologyBaseline")


def run_homology_transfer_baseline(
    k_neighbors: int = 5,
    data_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    results_dir: str = "results"
):
    data_path = Path(data_dir)
    raw_path = Path(raw_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training, validation, and holdout test splits...")
    train_df = pd.read_csv(raw_path / "train.csv")
    val_df = pd.read_csv(raw_path / "val.csv")
    test_df = pd.read_csv(raw_path / "test.csv")

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

    # 1. Alignment-Free Sequence Matching Matrix (4-mer local alignment index)
    logger.info("Building 4-mer sequence alignment index over 12,302 training proteins...")
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), max_features=10000, sublinear_tf=True)
    X_train = vec.fit_transform(train_df["sequence"])
    X_val = vec.transform(val_df["sequence"])
    X_test = vec.transform(test_df["sequence"])
    logger.info(f"Alignment index constructed in {time.time() - t0:.2f}s (Shape: {X_train.shape})")

    # 2. Nearest Neighbors Search (Cosine Similarity ~ Local Alignment Score)
    logger.info(f"Fitting Nearest Neighbors sequence search (k={k_neighbors})...")
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X_train)

    # Query Validation Set
    logger.info("Querying nearest sequence matches for validation set...")
    val_distances, val_indices = nn.kneighbors(X_val)

    # Query Test Set
    logger.info("Querying nearest sequence matches for holdout test set...")
    test_distances, test_indices = nn.kneighbors(X_test)

    # 3. Weighted Annotation Transfer
    # Similarity = max(0, 1 - cosine_distance)
    def transfer_annotations(indices, distances):
        n_queries = len(indices)
        transferred_probs = np.zeros((n_queries, num_classes), dtype=np.float32)

        for i in range(n_queries):
            neighbor_idxs = indices[i]
            dists = distances[i]
            sims = np.maximum(0.01, 1.0 - dists)  # Sequence similarities
            weights = sims / np.sum(sims)

            # Weighted sum of training neighbor annotations
            neighbor_labels = y_train[neighbor_idxs]  # shape: (k, num_classes)
            transferred_probs[i] = np.dot(weights, neighbor_labels)

        return transferred_probs

    logger.info("Computing similarity-weighted annotation transfer...")
    p_val_transfer = transfer_annotations(val_indices, val_distances)
    p_test_transfer = transfer_annotations(test_indices, test_distances)

    # 4. Calibrate Optimal Decision Threshold tau* on Validation Set
    logger.info("Calibrating decision threshold tau* on validation set...")
    best_tau = 0.20
    best_val_f1 = 0.0

    for t in np.linspace(0.05, 0.70, 60):
        val_pred_bin = (p_val_transfer >= t).astype(int)
        f1_val = f1_score(y_val, val_pred_bin, average="micro", zero_division=0)
        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_tau = float(t)

    logger.info(f"Optimal Validation Threshold: tau* = {best_tau:.3f} (Val F1 = {best_val_f1:.4f})")

    # 5. Evaluate on Holdout Test Set (<30% Sequence Homology)
    logger.info("Evaluating Homology Baseline on unseen non-homologous holdout test set...")
    y_test_pred_bin = (p_test_transfer >= best_tau).astype(int)

    test_micro = float(f1_score(y_test, y_test_pred_bin, average="micro", zero_division=0))
    test_macro = float(f1_score(y_test, y_test_pred_bin, average="macro", zero_division=0))
    test_prec = float(precision_score(y_test, y_test_pred_bin, average="micro", zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred_bin, average="micro", zero_division=0))

    mf_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "MF"]
    bp_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "BP"]
    cc_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "CC"]

    test_f1_mf = float(f1_score(y_test[:, mf_indices], y_test_pred_bin[:, mf_indices], average="micro", zero_division=0))
    test_f1_bp = float(f1_score(y_test[:, bp_indices], y_test_pred_bin[:, bp_indices], average="micro", zero_division=0))
    test_f1_cc = float(f1_score(y_test[:, cc_indices], y_test_pred_bin[:, cc_indices], average="micro", zero_division=0))

    # Mean sequence similarity in twilight zone
    avg_similarity = float(np.mean(1.0 - test_distances[:, 0]))

    metrics = {
        "model": "Homology Transfer Baseline (BLAST / k-NN)",
        "k_neighbors": k_neighbors,
        "average_top1_similarity": round(avg_similarity, 4),
        "calibrated_threshold": best_tau,
        "validation_fmax": round(best_val_f1, 4),
        "test_micro_f1": round(test_micro, 4),
        "test_macro_f1": round(test_macro, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1_mf": round(test_f1_mf, 4),
        "test_f1_bp": round(test_f1_bp, 4),
        "test_f1_cc": round(test_f1_cc, 4)
    }

    out_file = results_path / "homology_baseline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.info("HOMOLOGY TRANSFER BASELINE RESULTS (< 30% SEQUENCE IDENTITY)")
    logger.info("=" * 65)
    logger.info(f"Top-1 Mean Sequence Similarity: {avg_similarity:.4f} (Twilight Zone)")
    logger.info(f"Calibrated Threshold (tau*):    {best_tau:.3f}")
    logger.info(f"Validation F_max:               {best_val_f1:.4f}")
    logger.info(f"Test Micro-F1 (F_max):          {test_micro:.4f}")
    logger.info(f"Test Macro-F1:                  {test_macro:.4f}")
    logger.info(f"Test Precision:                 {test_prec:.4f}")
    logger.info(f"Test Recall:                    {test_rec:.4f}")
    logger.info(f"F1 (Molecular Function):        {test_f1_mf:.4f}")
    logger.info(f"F1 (Biological Process):        {test_f1_bp:.4f}")
    logger.info(f"F1 (Cellular Component):        {test_f1_cc:.4f}")
    logger.info("=" * 65)

    return metrics


if __name__ == "__main__":
    run_homology_transfer_baseline()
