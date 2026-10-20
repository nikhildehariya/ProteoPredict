"""
ProteoPredict - EXP-006: Pre-Trained ESM-2 + Biophysical Deep Hybrid Architecture
Fuses 320D pre-trained ESM-2 evolutionary embeddings with 430D biophysical descriptors
under Multi-Label Focal Loss (gamma=2.0, alpha=0.35) and strict CAFA validation calibration.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.ESMHybrid")


def get_focal_loss(gamma: float = 2.0, alpha: float = 0.35):
    """Multi-label Focal Loss to focus on rare, hard GO terms."""
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


def train_esm_hybrid(epochs: int = 15, batch_size: int = 64):
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Labels
    train_npz = np.load(processed_dir / "train_data.npz")
    val_npz = np.load(processed_dir / "val_data.npz")
    test_npz = np.load(processed_dir / "test_data.npz")
    y_train, y_val, y_test = train_npz["y"], val_npz["y"], test_npz["y"]

    with open(processed_dir / "go_info.json", "r") as f:
        go_info = json.load(f)
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})
    num_classes = len(classes)

    # 2. Load ESM-2 Embeddings (320D)
    esm_train = np.load(processed_dir / "esm2_train.npy")
    esm_val = np.load(processed_dir / "esm2_val.npy")
    esm_test = np.load(processed_dir / "esm2_test.npy")

    # 3. Load Biophysical Descriptors (430D)
    bio_train = np.load(processed_dir / "biophysical_train.npy")
    bio_val = np.load(processed_dir / "biophysical_val.npy")
    bio_test = np.load(processed_dir / "biophysical_test.npy")

    scaler = StandardScaler()
    bio_train_scaled = scaler.fit_transform(bio_train)
    bio_val_scaled = scaler.transform(bio_val)
    bio_test_scaled = scaler.transform(bio_test)

    # 4. Multi-Modal Feature Concatenation (320D ESM + 430D Bio = 750D)
    X_train = np.hstack([esm_train, bio_train_scaled]).astype(np.float32)
    X_val = np.hstack([esm_val, bio_val_scaled]).astype(np.float32)
    X_test = np.hstack([esm_test, bio_test_scaled]).astype(np.float32)

    logger.info(f"Loaded Multi-Modal ESM-2 + Biophysical Dataset: Shape={X_train.shape}")

    # 5. Build Deep Fusion Network
    inputs = Input(shape=(X_train.shape[1],), name="esm_bio_input")
    x = Dense(512, activation="relu", name="dense1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Dropout(0.4, name="drop1")(x)

    x = Dense(256, activation="relu", name="dense2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Dropout(0.3, name="drop2")(x)

    x = Dense(128, activation="relu", name="dense3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = Dropout(0.2, name="drop3")(x)

    outputs = Dense(num_classes, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ProteoPredict_ESM2_Biophysical_Hybrid")
    model.summary(print_fn=logger.info)

    loss_fn = get_focal_loss(gamma=2.0, alpha=0.35)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn)

    best_weights_path = models_dir / "esm2_biophysical_hybrid_best.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(best_weights_path), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

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

    # Validation Calibration
    val_probs = model.predict(X_val, batch_size=batch_size, verbose=0)
    opt_threshold, val_fmax = calibrate_validation_threshold(y_val, val_probs)
    logger.info(f"Validation F_max: {val_fmax:.4f} at calibrated threshold tau* = {opt_threshold:.3f}")

    # Unseen Test Evaluation
    test_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_metrics = evaluate_with_threshold(y_test, test_probs, opt_threshold, go_metadata, classes)
    test_metrics["model"] = "ESM-2 + Biophysical Deep Hybrid (EXP-006)"
    test_metrics["validation_fmax"] = val_fmax
    test_metrics["training_time_s"] = round(training_time, 1)

    logger.info("\n" + "=" * 55)
    logger.info("FINAL BENCHMARK: ESM-2 + BIOPHYSICAL DEEP HYBRID")
    logger.info("=" * 55)
    logger.info(f"Optimal Threshold: {test_metrics['calibrated_threshold']:.3f}")
    logger.info(f"Test Precision:    {test_metrics['precision_micro']:.4f}")
    logger.info(f"Test Recall:       {test_metrics['recall_micro']:.4f}")
    logger.info(f"Test Micro-F1:     {test_metrics['f1_micro']:.4f}")
    logger.info(f"Test Macro-F1:     {test_metrics['f1_macro']:.4f}")
    logger.info(f"F1 (Molecular Function):  {test_metrics.get('f1_mf', 0):.4f}")
    logger.info(f"F1 (Biological Process):  {test_metrics.get('f1_bp', 0):.4f}")
    logger.info(f"F1 (Cellular Component):  {test_metrics.get('f1_cc', 0):.4f}")
    logger.info("=" * 55)

    with open(results_dir / "esm2_hybrid_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    train_esm_hybrid()
