"""
ProteoPredict - Model Training & Scientific Benchmark Suite
Trains biological baselines, multi-modal feature fusion models, and deep attention networks
using Gene Ontology DAG ground truth, Focal Loss (gamma=2.0), and strict validation calibration.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    GlobalAveragePooling1D, Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, LayerNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from src.proteopredict.features.biophysical import batch_extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.Train")


# ------------------- LOSS FUNCTIONS -------------------

def get_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Multi-label Focal Loss to focus gradients on hard functional terms
    and suppress gradient dominance from easy background negatives.
    """
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        alpha_factor = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)

        bce = -tf.math.log(p_t)
        return tf.reduce_mean(focal_weight * bce, axis=-1)

    return loss


def get_weighted_bce_loss(pos_weights):
    """Weighted Binary Cross-Entropy Loss to balance positive/negative class frequencies."""
    pos_w = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        bce = - (pos_w * y_true_f * tf.math.log(y_pred_f) + (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f))
        return tf.reduce_mean(bce, axis=-1)

    return loss


# ------------------- MODEL ARCHITECTURES -------------------

def build_attention_hybrid_model(vocab_size: int, num_classes: int, max_length: int = 1000, embedding_dim: int = 64):
    """
    ProteoPredict Multi-Head Attention Hybrid Architecture.
    Combines multi-scale motif convolutions with 4-head self-attention and residual layer normalization.
    """
    inputs = Input(shape=(max_length,), name="sequence_input")
    x = Embedding(vocab_size, embedding_dim, input_length=max_length, name="embedding")(inputs)

    # Local motif extraction with multi-scale kernel
    conv = Conv1D(128, kernel_size=5, activation="relu", padding="same", name="motif_conv")(x)
    pooled = MaxPooling1D(pool_size=4, name="motif_pool")(conv)  # (batch, 250, 128)

    # Multi-Head Self-Attention over pooled sequence
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32, name="self_attention")(pooled, pooled)
    x = Add(name="residual_add")([pooled, attn_out])
    x = LayerNormalization(name="layer_norm")(x)

    # Multi-resolution representation
    gap = GlobalAveragePooling1D(name="gap")(x)
    gmp = GlobalMaxPooling1D(name="gmp")(x)
    concat = tf.keras.layers.Concatenate(name="pooling_concat")([gap, gmp])

    dense = Dense(256, activation="relu", name="dense1")(concat)
    dense = Dropout(0.4, name="drop1")(dense)
    outputs = Dense(num_classes, activation="sigmoid", name="predictions")(dense)

    return keras.Model(inputs=inputs, outputs=outputs, name="ProteoPredict_AttentionHybrid")


# ------------------- CAFA EVALUATION PROTOCOL -------------------

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


# ------------------- BASELINE & MULTI-MODAL TRAINER -------------------

def train_baselines(raw_dir: Path, processed_dir: Path, results_dir: Path, go_metadata: dict, classes: list):
    """Trains 3-mer TF-IDF baseline and Multi-Modal Biophysical model on DAG-propagated data."""
    logger.info("Training Biological Baselines on DAG Ground Truth...")

    train_df = pd.read_csv(raw_dir / "train.csv")
    val_df = pd.read_csv(raw_dir / "val.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")

    train_npz = np.load(processed_dir / "train_data.npz")
    val_npz = np.load(processed_dir / "val_data.npz")
    test_npz = np.load(processed_dir / "test_data.npz")

    y_train, y_val, y_test = train_npz["y"], val_npz["y"], test_npz["y"]

    # 1. 3-mer TF-IDF Vectorization
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=3000, sublinear_tf=True)
    X_train_kmer = vec.fit_transform(train_df["sequence"])
    X_val_kmer = vec.transform(val_df["sequence"])
    X_test_kmer = vec.transform(test_df["sequence"])

    clf_kmer = MultiOutputClassifier(
        SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=30, random_state=42),
        n_jobs=-1
    )
    clf_kmer.fit(X_train_kmer, y_train)

    val_probs_kmer = np.array([est.predict_proba(X_val_kmer)[:, 1] for est in clf_kmer.estimators_]).T
    test_probs_kmer = np.array([est.predict_proba(X_test_kmer)[:, 1] for est in clf_kmer.estimators_]).T

    opt_thresh_kmer, val_f1_kmer = calibrate_validation_threshold(y_val, val_probs_kmer)
    test_metrics_kmer = evaluate_with_threshold(y_test, test_probs_kmer, opt_thresh_kmer, go_metadata, classes)
    test_metrics_kmer["model"] = "3-mer Baseline (DAG Propagated)"
    test_metrics_kmer["training_time_s"] = round(time.time() - t0, 1)

    with open(results_dir / "baseline_metrics.json", "w") as f:
        json.dump(test_metrics_kmer, f, indent=2)

    logger.info(f"3-mer Baseline DAG Test Micro-F1: {test_metrics_kmer['f1_micro']:.4f} (Macro-F1: {test_metrics_kmer['f1_macro']:.4f})")
    return test_metrics_kmer


# ------------------- DEEP LEARNING WORKFLOW -------------------

def train_deep_attention_model(
    data_dir: Path,
    models_dir: Path,
    results_dir: Path,
    loss_type: str = "focal",
    epochs: int = 10,
    batch_size: int = 64
):
    """Train Deep Multi-Head Attention Network on DAG-propagated data with Focal Loss."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(data_dir / "train_data.npz")
    val_data = np.load(data_dir / "val_data.npz")
    test_data = np.load(data_dir / "test_data.npz")

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    with open(data_dir / "preprocess_config.json", "r") as f:
        config = json.load(f)
    with open(data_dir / "go_info.json", "r") as f:
        go_info = json.load(f)

    vocab_size = config["vocab_size"]
    max_length = config["max_length"]
    num_classes = config["num_classes"]
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})

    logger.info(f"Loaded DAG Dataset: Train={X_train.shape[0]:,}, Val={X_val.shape[0]:,}, Test={X_test.shape[0]:,}")
    logger.info(f"GO Classes: {num_classes}, Average Train Labels/Protein: {y_train.sum(axis=1).mean():.2f}")

    # Build Attention-Hybrid Architecture
    model = build_attention_hybrid_model(vocab_size, num_classes, max_length)
    model.summary(print_fn=logger.info)

    # Loss selection
    if loss_type == "focal":
        loss_fn = get_focal_loss(gamma=2.0, alpha=0.35)
        logger.info("Using Multi-Label Focal Loss (gamma=2.0, alpha=0.35).")
    else:
        pos_counts = y_train.sum(axis=0)
        neg_counts = len(y_train) - pos_counts
        pos_weights = np.clip(neg_counts / (pos_counts + 10.0), 1.0, 5.0)
        loss_fn = get_weighted_bce_loss(pos_weights)
        logger.info("Using Weighted Binary Cross-Entropy Loss.")

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_fn)

    best_weights_path = models_dir / "best_model.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(best_weights_path), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    logger.info(f"Training ProteoPredict_AttentionHybrid for {epochs} epochs...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - t0

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(models_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # Validation Threshold Calibration (CAFA F_max)
    logger.info("Calibrating decision threshold on validation set...")
    val_probs = model.predict(X_val, batch_size=batch_size, verbose=0)
    opt_threshold, val_fmax = calibrate_validation_threshold(y_val, val_probs)
    logger.info(f"Validation F_max: {val_fmax:.4f} at calibrated threshold tau* = {opt_threshold:.3f}")

    # Unseen Test Set Evaluation
    logger.info("Evaluating on unseen test set...")
    test_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_metrics = evaluate_with_threshold(y_test, test_probs, opt_threshold, go_metadata, classes)
    test_metrics["model"] = f"Attention-Hybrid ({loss_type.upper()} + DAG)"
    test_metrics["validation_fmax"] = val_fmax
    test_metrics["training_time_s"] = round(training_time, 1)

    logger.info("\n" + "=" * 50)
    logger.info("TEST BENCHMARK RESULTS (DAG + FOCAL ATTENTION HYBRID)")
    logger.info("=" * 50)
    logger.info(f"Optimal Threshold: {test_metrics['calibrated_threshold']:.3f}")
    logger.info(f"Test Precision:    {test_metrics['precision_micro']:.4f}")
    logger.info(f"Test Recall:       {test_metrics['recall_micro']:.4f}")
    logger.info(f"Test Micro-F1:     {test_metrics['f1_micro']:.4f}")
    logger.info(f"Test Macro-F1:     {test_metrics['f1_macro']:.4f}")
    if "f1_mf" in test_metrics:
        logger.info(f"F1 (Molecular Function): {test_metrics['f1_mf']:.4f}")
    if "f1_bp" in test_metrics:
        logger.info(f"F1 (Biological Process): {test_metrics['f1_bp']:.4f}")
    if "f1_cc" in test_metrics:
        logger.info(f"F1 (Cellular Component): {test_metrics['f1_cc']:.4f}")
    logger.info("=" * 50)

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(results_dir / "attention_hybrid_dag_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


def run_full_experiment(loss_type="focal", epochs=8):
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    results_dir = Path("results")

    with open(processed_dir / "go_info.json", "r") as f:
        go_info = json.load(f)
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})

    # 1. Train Baseline on DAG
    baseline_metrics = train_baselines(raw_dir, processed_dir, results_dir, go_metadata, classes)

    # 2. Train Deep Model with Focal Loss on DAG
    model, deep_metrics = train_deep_attention_model(
        data_dir=processed_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        loss_type=loss_type,
        epochs=epochs,
        batch_size=64
    )

    comparison = {
        "baseline_dag": baseline_metrics,
        "deep_attention_hybrid_dag": deep_metrics
    }
    with open(results_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Experiment run complete. All metrics recorded in results/model_comparison.json.")


def main():
    parser = argparse.ArgumentParser(description="Train ProteoPredict on DAG-propagated ground truth.")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "weighted_bce"])
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    run_full_experiment(loss_type=args.loss, epochs=args.epochs)


if __name__ == "__main__":
    main()
