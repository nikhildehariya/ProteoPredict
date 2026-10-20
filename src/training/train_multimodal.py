"""
ProteoPredict - EXP-004: Multi-Modal Biophysical Fusion Training Suite
Fuses 3-mer TF-IDF sequence representations with 430-dimensional
biochemical descriptors (AAC, DPC, physicochemical properties)
to predict Gene Ontology functions under strict validation calibration.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.proteopredict.features.biophysical import batch_extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.MultiModal")


def get_focal_loss(gamma: float = 2.0, alpha: float = 0.35):
    """Multi-label Focal Loss."""
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        alpha_factor = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)
        bce = -tf.math.log(p_t)
        return tf.reduce_mean(focal_weight * bce, axis=-1)
    return loss


def calibrate_validation_threshold(y_true, y_pred_proba, num_steps: int = 60):
    """Calibrate optimal decision threshold tau* exclusively on the validation set."""
    thresholds = np.linspace(0.05, 0.70, num_steps)
    best_f1 = 0.0
    best_threshold = 0.25

    for t in thresholds:
        y_bin = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_bin, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, float(best_f1)


def evaluate_with_threshold(y_true, y_pred_proba, threshold: float, go_metadata: dict = None, classes: list = None):
    """Evaluate predictions on unseen test partition using fixed validation-calibrated threshold."""
    y_pred_bin = (y_pred_proba >= threshold).astype(int)

    precision_micro = float(precision_score(y_true, y_pred_bin, average="micro", zero_division=0))
    recall_micro = float(recall_score(y_true, y_pred_bin, average="micro", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))

    metrics = {
        "calibrated_threshold": threshold,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }

    if go_metadata and classes:
        mf_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "MF"]
        bp_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "BP"]
        cc_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "CC"]

        if mf_indices:
            metrics["f1_mf"] = float(f1_score(y_true[:, mf_indices], y_pred_bin[:, mf_indices], average="micro", zero_division=0))
        if bp_indices:
            metrics["f1_bp"] = float(f1_score(y_true[:, bp_indices], y_pred_bin[:, bp_indices], average="micro", zero_division=0))
        if cc_indices:
            metrics["f1_cc"] = float(f1_score(y_true[:, cc_indices], y_pred_bin[:, cc_indices], average="micro", zero_division=0))

    return metrics


def get_or_extract_biophysical(df: pd.DataFrame, cache_path: Path) -> np.ndarray:
    """Load cached biophysical descriptors or extract and cache."""
    if cache_path.exists():
        logger.info(f"Loading cached biophysical features from {cache_path}...")
        return np.load(cache_path)
    logger.info(f"Extracting 430D biophysical features for {len(df)} sequences...")
    feats = batch_extract_features(df["sequence"].tolist())
    np.save(cache_path, feats)
    logger.info(f"Cached biophysical features to {cache_path} (shape: {feats.shape})")
    return feats


def run_multimodal_experiment():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(raw_dir / "train.csv")
    val_df = pd.read_csv(raw_dir / "val.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")

    train_npz = np.load(processed_dir / "train_data.npz")
    val_npz = np.load(processed_dir / "val_data.npz")
    test_npz = np.load(processed_dir / "test_data.npz")

    y_train, y_val, y_test = train_npz["y"], val_npz["y"], test_npz["y"]

    with open(processed_dir / "go_info.json", "r") as f:
        go_info = json.load(f)
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})
    num_classes = len(classes)

    # 1. 3-mer TF-IDF Vectorizer
    logger.info("Computing 3-mer TF-IDF representations (3,000D)...")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=3000, sublinear_tf=True)
    X_train_kmer = vec.fit_transform(train_df["sequence"])
    X_val_kmer = vec.transform(val_df["sequence"])
    X_test_kmer = vec.transform(test_df["sequence"])

    # 2. 430D Biophysical Descriptors
    bio_train = get_or_extract_biophysical(train_df, processed_dir / "biophysical_train.npy")
    bio_val = get_or_extract_biophysical(val_df, processed_dir / "biophysical_val.npy")
    bio_test = get_or_extract_biophysical(test_df, processed_dir / "biophysical_test.npy")

    scaler = StandardScaler()
    bio_train_scaled = scaler.fit_transform(bio_train)
    bio_val_scaled = scaler.transform(bio_val)
    bio_test_scaled = scaler.transform(bio_test)

    # 3. Multi-Modal Fusion (3,000 k-mers + 430 biophysical = 3,430 dimensions)
    logger.info("Concatenating multi-modal feature matrices (3,430D)...")
    X_train_fused = hstack([X_train_kmer, csr_matrix(bio_train_scaled)]).tocsr()
    X_val_fused = hstack([X_val_kmer, csr_matrix(bio_val_scaled)]).tocsr()
    X_test_fused = hstack([X_test_kmer, csr_matrix(bio_test_scaled)]).tocsr()
    logger.info(f"Fused representations shape: {X_train_fused.shape}")

    # Model 1: Multi-Modal Linear SGD Classifier
    logger.info("\n--- Training Multi-Modal Linear Classifier (EXP-004a) ---")
    t0 = time.time()
    clf = MultiOutputClassifier(
        SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=35, random_state=42),
        n_jobs=-1
    )
    clf.fit(X_train_fused, y_train)
    t_train = time.time() - t0

    val_probs_linear = np.array([est.predict_proba(X_val_fused)[:, 1] for est in clf.estimators_]).T
    test_probs_linear = np.array([est.predict_proba(X_test_fused)[:, 1] for est in clf.estimators_]).T

    opt_thresh_linear, val_fmax_linear = calibrate_validation_threshold(y_val, val_probs_linear)
    linear_metrics = evaluate_with_threshold(y_test, test_probs_linear, opt_thresh_linear, go_metadata, classes)
    linear_metrics["model"] = "Multi-Modal (3-mer + 430D Biophysical Linear)"
    linear_metrics["validation_fmax"] = val_fmax_linear
    linear_metrics["training_time_s"] = round(t_train, 1)

    logger.info(f"Multi-Modal Linear Test Micro-F1: {linear_metrics['f1_micro']:.4f} (Macro-F1: {linear_metrics['f1_macro']:.4f})")
    logger.info(f"MF: {linear_metrics.get('f1_mf', 0):.4f} | BP: {linear_metrics.get('f1_bp', 0):.4f} | CC: {linear_metrics.get('f1_cc', 0):.4f}")

    # Model 2: Multi-Modal Deep Neural Network (MLP with Focal Loss)
    logger.info("\n--- Training Multi-Modal Deep Fusion Network (EXP-004b) ---")
    input_dim = X_train_fused.shape[1]

    # Convert sparse to dense batch generator or dense array for moderate dimensions
    # 12,302 x 3430 float32 is ~168 MB in memory - easily fits in RAM!
    X_train_dense = X_train_fused.toarray().astype(np.float32)
    X_val_dense = X_val_fused.toarray().astype(np.float32)
    X_test_dense = X_test_fused.toarray().astype(np.float32)

    inputs = Input(shape=(input_dim,), name="fused_input")
    x = Dense(512, activation="relu", name="dense_1")(inputs)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.4, name="drop_1")(x)

    x = Dense(256, activation="relu", name="dense_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.3, name="drop_2")(x)

    outputs = Dense(num_classes, activation="sigmoid", name="predictions")(x)
    nn_model = keras.Model(inputs=inputs, outputs=outputs, name="ProteoPredict_MultiModal_MLP")

    loss_fn = get_focal_loss(gamma=2.0, alpha=0.35)
    nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(models_dir / "multimodal_mlp_best.h5"), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    t0 = time.time()
    nn_model.fit(
        X_train_dense, y_train,
        validation_data=(X_val_dense, y_val),
        epochs=12,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    t_nn = time.time() - t0

    val_probs_nn = nn_model.predict(X_val_dense, batch_size=64, verbose=0)
    opt_thresh_nn, val_fmax_nn = calibrate_validation_threshold(y_val, val_probs_nn)
    test_probs_nn = nn_model.predict(X_test_dense, batch_size=64, verbose=0)
    nn_metrics = evaluate_with_threshold(y_test, test_probs_nn, opt_thresh_nn, go_metadata, classes)
    nn_metrics["model"] = "Multi-Modal Deep Fusion Network (MLP + Focal Loss)"
    nn_metrics["validation_fmax"] = val_fmax_nn
    nn_metrics["training_time_s"] = round(t_nn, 1)

    logger.info(f"Multi-Modal Deep MLP Test Micro-F1: {nn_metrics['f1_micro']:.4f} (Macro-F1: {nn_metrics['f1_macro']:.4f})")
    logger.info(f"MF: {nn_metrics.get('f1_mf', 0):.4f} | BP: {nn_metrics.get('f1_bp', 0):.4f} | CC: {nn_metrics.get('f1_cc', 0):.4f}")

    results = {
        "multimodal_linear": linear_metrics,
        "multimodal_deep_mlp": nn_metrics
    }
    with open(results_dir / "multimodal_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Multi-modal experiment complete! Results saved to results/multimodal_metrics.json.")
    return results


if __name__ == "__main__":
    run_multimodal_experiment()
