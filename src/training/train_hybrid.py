"""
ProteoPredict - EXP-005: Deep ResNet-BiLSTM Multi-Head Attention Fusion Architecture
Combines multi-scale motif convolutions (k=3, 5, 9), Bidirectional LSTM,
4-Head Self-Attention, and 430D biophysical profile under Multi-Label Focal Loss.
Strictly evaluated on unseen test set using validation-calibrated threshold.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    GlobalAveragePooling1D, Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, LayerNormalization, MultiHeadAttention, Add, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.proteopredict.features.biophysical import batch_extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.DeepHybrid")


# ------------------- FOCAL LOSS -------------------

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


# ------------------- ARCHITECTURE -------------------

def build_resnet_bilstm_attention_model(vocab_size: int, num_classes: int, max_length: int = 1000, bio_dim: int = 430):
    """
    Constructs the ProteoPredict Deep Multi-Modal Hybrid Architecture:
    Branch 1 (Sequence):
      - Embedding(64D)
      - Multi-Scale Parallel Convolutions (k=3, 5, 9)
      - Residual Add + LayerNorm
      - MaxPool(pool_size=4) -> L=250
      - Bidirectional LSTM (hidden=64)
      - 4-Head Self-Attention + Residual Add + LayerNorm
      - Concatenation of GAP and GMP
    Branch 2 (Biophysical):
      - 430D Descriptors -> Dense(128) -> BatchNorm -> Dropout(0.2)
    Fusion Head:
      - Concatenate(Branch 1, Branch 2)
      - Dense(256) -> BatchNorm -> Dropout(0.3)
      - Output Dense(75, sigmoid)
    """
    # 1. Sequence Branch
    seq_input = Input(shape=(max_length,), name="seq_input")
    emb = Embedding(vocab_size, 64, input_length=max_length, name="seq_embedding")(seq_input)

    # Multi-scale motif convolutions
    conv3 = Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv_k3")(emb)
    conv5 = Conv1D(64, kernel_size=5, padding="same", activation="relu", name="conv_k5")(emb)
    conv9 = Conv1D(64, kernel_size=9, padding="same", activation="relu", name="conv_k9")(emb)
    multi_conv = Concatenate(name="multi_scale_concat")([conv3, conv5, conv9])  # 192 channels

    # Residual projection
    res_proj = Conv1D(192, kernel_size=1, padding="same", name="res_proj")(emb)
    res_out = Add(name="conv_res_add")([multi_conv, res_proj])
    res_out = LayerNormalization(name="conv_ln")(res_out)

    pooled = MaxPooling1D(pool_size=4, name="seq_pool")(res_out)  # L=250, channels=192

    # Bidirectional LSTM for sequential syntax
    bilstm = Bidirectional(LSTM(48, return_sequences=True), name="bilstm")(pooled)  # (batch, 250, 96)

    # Multi-Head Self-Attention for long-range interactions
    attn = MultiHeadAttention(num_heads=4, key_dim=24, name="self_attn")(bilstm, bilstm)
    attn_res = Add(name="attn_res_add")([bilstm, attn])
    attn_ln = LayerNormalization(name="attn_ln")(attn_res)

    gap = GlobalAveragePooling1D(name="seq_gap")(attn_ln)
    gmp = GlobalMaxPooling1D(name="seq_gmp")(attn_ln)
    seq_rep = Concatenate(name="seq_rep_concat")([gap, gmp])  # 192D

    # 2. Biophysical Branch
    bio_input = Input(shape=(bio_dim,), name="bio_input")
    bio_d = Dense(128, activation="relu", name="bio_dense1")(bio_input)
    bio_d = BatchNormalization(name="bio_bn1")(bio_d)
    bio_d = Dropout(0.2, name="bio_drop1")(bio_d)
    bio_rep = Dense(64, activation="relu", name="bio_dense2")(bio_d)

    # 3. Multi-Modal Fusion
    fused = Concatenate(name="fusion_concat")([seq_rep, bio_rep])
    dense_f1 = Dense(256, activation="relu", name="fusion_dense1")(fused)
    dense_f1 = BatchNormalization(name="fusion_bn1")(dense_f1)
    dense_f1 = Dropout(0.3, name="fusion_drop1")(dense_f1)

    outputs = Dense(num_classes, activation="sigmoid", name="predictions")(dense_f1)

    model = keras.Model(inputs=[seq_input, bio_input], outputs=outputs, name="ProteoPredict_ResNet_BiLSTM_Attention")
    return model


# ------------------- CALIBRATION & EVALUATION -------------------

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


# ------------------- MAIN TRAINING WORKFLOW -------------------

def run_hybrid_experiment(epochs: int = 8, batch_size: int = 64):
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load sequence arrays
    train_npz = np.load(processed_dir / "train_data.npz")
    val_npz = np.load(processed_dir / "val_data.npz")
    test_npz = np.load(processed_dir / "test_data.npz")

    X_train_seq, y_train = train_npz["X"], train_npz["y"]
    X_val_seq, y_val = val_npz["X"], val_npz["y"]
    X_test_seq, y_test = test_npz["X"], test_npz["y"]

    # 2. Load pre-extracted biophysical features
    bio_train = np.load(processed_dir / "biophysical_train.npy")
    bio_val = np.load(processed_dir / "biophysical_val.npy")
    bio_test = np.load(processed_dir / "biophysical_test.npy")

    scaler = StandardScaler()
    bio_train_scaled = scaler.fit_transform(bio_train)
    bio_val_scaled = scaler.transform(bio_val)
    bio_test_scaled = scaler.transform(bio_test)

    with open(processed_dir / "preprocess_config.json", "r") as f:
        config = json.load(f)
    with open(processed_dir / "go_info.json", "r") as f:
        go_info = json.load(f)

    vocab_size = config["vocab_size"]
    max_length = config["max_length"]
    num_classes = config["num_classes"]
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})

    logger.info("Building Deep ResNet-BiLSTM Multi-Head Attention Hybrid...")
    model = build_resnet_bilstm_attention_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_length=max_length,
        bio_dim=bio_train.shape[1]
    )
    model.summary(print_fn=logger.info)

    loss_fn = get_focal_loss(gamma=2.0, alpha=0.35)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn)

    best_weights_path = models_dir / "resnet_bilstm_attn_best.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(best_weights_path), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    logger.info(f"Training ResNet-BiLSTM-Attention Hybrid for {epochs} epochs...")
    t0 = time.time()
    model.fit(
        [X_train_seq, bio_train_scaled], y_train,
        validation_data=([X_val_seq, bio_val_scaled], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - t0

    # Validation Calibration
    logger.info("Calibrating decision threshold on validation set...")
    val_probs = model.predict([X_val_seq, bio_val_scaled], batch_size=batch_size, verbose=0)
    opt_threshold, val_fmax = calibrate_validation_threshold(y_val, val_probs)
    logger.info(f"Validation F_max: {val_fmax:.4f} at tau* = {opt_threshold:.3f}")

    # Test Evaluation
    logger.info("Evaluating on unseen test set...")
    test_probs = model.predict([X_test_seq, bio_test_scaled], batch_size=batch_size, verbose=0)
    test_metrics = evaluate_with_threshold(y_test, test_probs, opt_threshold, go_metadata, classes)
    test_metrics["model"] = "ResNet-BiLSTM-Attention Hybrid (EXP-005)"
    test_metrics["validation_fmax"] = val_fmax
    test_metrics["training_time_s"] = round(training_time, 1)

    logger.info("\n" + "=" * 55)
    logger.info("TEST RESULTS: ResNet-BiLSTM-Attention Hybrid (EXP-005)")
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

    with open(results_dir / "resnet_bilstm_attn_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    run_hybrid_experiment(epochs=8)
