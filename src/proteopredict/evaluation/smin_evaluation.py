"""
ProteoPredict - CAFA Semantic Distance (S_min) Evaluation Suite
Computes Remaining Uncertainty (RU), Misinformation (MI), and Minimum Semantic Distance:
S_min = min_tau sqrt(RU(tau)^2 + MI(tau)^2)
weighted by Shannon Information Accretion IA(c) = -log2(P(c)).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.SMin")


def compute_smin_for_probs(y_true, y_pred_probs, ia_vector, num_thresholds: int = 60):
    """
    Computes S_min = min_tau sqrt(RU(tau)^2 + MI(tau)^2)
    where RU and MI are weighted by Information Accretion.
    """
    thresholds = np.linspace(0.02, 0.80, num_thresholds)
    best_smin = float("inf")
    best_tau = 0.30
    best_ru = 0.0
    best_mi = 0.0

    N = len(y_true)

    for tau in thresholds:
        y_bin = (y_pred_probs >= tau).astype(int)

        # False Negatives: True but not predicted
        fn = (y_true == 1) & (y_bin == 0)
        # False Positives: Predicted but not true
        fp = (y_true == 0) & (y_bin == 1)

        # Weighted by IA
        ru = np.sum(fn * ia_vector) / N
        mi = np.sum(fp * ia_vector) / N

        dist = np.sqrt(ru**2 + mi**2)
        if dist < best_smin:
            best_smin = dist
            best_tau = float(tau)
            best_ru = float(ru)
            best_mi = float(mi)

    return {
        "s_min": round(best_smin, 4),
        "optimal_tau": round(best_tau, 3),
        "remaining_uncertainty": round(best_ru, 4),
        "misinformation": round(best_mi, 4)
    }


def evaluate_all_smin():
    data_path = Path("data/processed")
    results_path = Path("results")

    # Load Labels
    test_data = np.load(data_path / "test_data.npz")
    y_test = test_data["y"]
    train_data = np.load(data_path / "train_data.npz")
    y_train = train_data["y"]

    # Load IA weights
    with open(data_path / "ia_weights.json", "r") as f:
        ia_dict = json.load(f)

    with open(data_path / "go_info.json", "r") as f:
        go_info = json.load(f)
    classes = go_info["classes"]

    ia_vector = np.array([ia_dict[c]["ia_weight"] for c in classes], dtype=np.float32)

    logger.info(f"Loaded IA weights for {len(ia_vector)} terms (Mean IA = {np.mean(ia_vector):.2f} bits)")

    # 1. EXP-000: Naive Marginal Baseline
    p_train = y_train.mean(axis=0)
    p_test_naive = np.tile(p_train, (len(y_test), 1))
    smin_naive = compute_smin_for_probs(y_test, p_test_naive, ia_vector)
    logger.info(f"EXP-000 (Naive Prior):    S_min = {smin_naive['s_min']} (tau={smin_naive['optimal_tau']})")

    # 2. Homology Baseline
    from src.proteopredict.evaluation.homology_baseline import run_homology_transfer_baseline
    # Load test df and vec
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd

    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=3000, sublinear_tf=True)
    X_tr = vec.fit_transform(train_df["sequence"])
    X_te = vec.transform(test_df["sequence"])
    nn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1).fit(X_tr)
    dists, idxs = nn.kneighbors(X_te)
    p_test_homology = np.zeros_like(y_test, dtype=np.float32)
    for i in range(len(test_df)):
        sims = np.maximum(0.01, 1.0 - dists[i])
        p_test_homology[i] = np.dot(sims / sims.sum(), y_train[idxs[i]])

    smin_homology = compute_smin_for_probs(y_test, p_test_homology, ia_vector)
    logger.info(f"Homology Baseline (k-NN): S_min = {smin_homology['s_min']} (tau={smin_homology['optimal_tau']})")

    # 3. EXP-006: ESM-2 Hybrid
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    esm_test = np.load(data_path / "esm2_test.npy")
    bio_train = np.load(data_path / "biophysical_train.npy")
    bio_test = np.load(data_path / "biophysical_test.npy")
    scaler = StandardScaler().fit(bio_train)
    X_test_esm = np.hstack([esm_test, scaler.transform(bio_test)]).astype(np.float32)

    model_esm = tf.keras.models.load_model("models/esm2_biophysical_hybrid_best.h5", compile=False)
    p_test_esm = model_esm.predict(X_test_esm, batch_size=128, verbose=0)
    smin_esm = compute_smin_for_probs(y_test, p_test_esm, ia_vector)
    logger.info(f"EXP-006 (ESM-2 Hybrid):   S_min = {smin_esm['s_min']} (tau={smin_esm['optimal_tau']})")

    # 4. EXP-007: Tri-Modal Multi-Branch Deep Network
    import pickle
    with open("models/kmer_vectorizer_1500.pkl", "rb") as f:
        vec_1500 = pickle.load(f)
    kmer_test = vec_1500.transform(test_df["sequence"]).toarray().astype(np.float32)
    bio_test_s = scaler.transform(bio_test).astype(np.float32)

    model_trimodal = tf.keras.models.load_model("models/trimodal_multibranch_best.h5", compile=False)
    p_mf, p_bp, p_cc = model_trimodal.predict([esm_test, kmer_test, bio_test_s], batch_size=128, verbose=0)

    with open("models/trimodal_metadata.json", "r") as f:
        meta = json.load(f)
    mf_indices = meta["mf_indices"]
    bp_indices = meta["bp_indices"]
    cc_indices = meta["cc_indices"]

    p_test_trimodal = np.zeros_like(y_test, dtype=np.float32)
    p_test_trimodal[:, mf_indices] = p_mf
    p_test_trimodal[:, bp_indices] = p_bp
    p_test_trimodal[:, cc_indices] = p_cc

    smin_trimodal = compute_smin_for_probs(y_test, p_test_trimodal, ia_vector)
    logger.info(f"EXP-007 (Tri-Modal Deep): S_min = {smin_trimodal['s_min']} (tau={smin_trimodal['optimal_tau']})")

    # Save summary
    all_smin = {
        "EXP-000_Naive": smin_naive,
        "Homology_Baseline": smin_homology,
        "EXP-006_ESM2_Hybrid": smin_esm,
        "EXP-007_TriModal_MultiBranch": smin_trimodal
    }

    out_file = results_path / "cafa_smin_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_smin, f, indent=2)

    logger.info(f"\nAll CAFA S_min results saved to {out_file}")
    return all_smin


if __name__ == "__main__":
    evaluate_all_smin()
