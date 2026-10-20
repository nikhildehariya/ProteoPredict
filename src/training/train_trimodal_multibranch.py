"""
ProteoPredict - EXP-007: Tri-Modal Multi-Branch Deep Fusion Network Architecture
Fuses:
  1. Meta AI ESM-2 (320D evolutionary embeddings)
  2. Alignment-Free 3-mer TF-IDF (1,500D sequence motifs)
  3. 430-Dimensional Biophysical Descriptors (AAC, DPC, charge, hydropathy)
into a shared representation feeding three specialized Sub-Ontology prediction heads:
  - Molecular Function (MF) Head
  - Biological Process (BP) Head
  - Cellular Component (CC) Head
under Multi-Label Focal Loss with per-ontology validation threshold calibration.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.TriModalMultiBranch")


def get_focal_loss(gamma: float = 2.0, alpha: float = 0.35):
    """Multi-label Focal Loss to address extreme positive/negative imbalance."""
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        alpha_factor = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)
        bce = -tf.math.log(p_t)
        return tf.reduce_mean(focal_weight * bce, axis=-1)
    return loss


def calibrate_branch_threshold(y_true, y_pred_proba, num_steps: int = 50):
    """Calibrate optimal decision threshold tau* exclusively on the validation set."""
    thresholds = np.linspace(0.10, 0.70, num_steps)
    best_f1 = 0.0
    best_threshold = 0.35

    for t in thresholds:
        y_bin = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_bin, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, float(best_f1)


def train_trimodal_multibranch(epochs: int = 20, batch_size: int = 64, kmer_features: int = 1500):
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits and Sequences
    logger.info("Loading dataset splits...")
    train_df = pd.read_csv(raw_dir / "train.csv")
    val_df = pd.read_csv(raw_dir / "val.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")

    train_npz = np.load(processed_dir / "train_data.npz")
    val_npz = np.load(processed_dir / "val_data.npz")
    test_npz = np.load(processed_dir / "test_data.npz")
    y_train = train_npz["y"]
    y_val = val_npz["y"]
    y_test = test_npz["y"]

    with open(processed_dir / "go_info.json", "r") as f:
        go_info = json.load(f)
    classes = go_info["classes"]
    go_metadata = go_info.get("term_metadata", {})

    # Separate class indices by sub-ontology
    mf_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "MF"]
    bp_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "BP"]
    cc_indices = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "CC"]

    logger.info(f"Sub-ontology partitioning: MF={len(mf_indices)}, BP={len(bp_indices)}, CC={len(cc_indices)} (Total: {len(classes)})")

    y_train_mf, y_val_mf, y_test_mf = y_train[:, mf_indices], y_val[:, mf_indices], y_test[:, mf_indices]
    y_train_bp, y_val_bp, y_test_bp = y_train[:, bp_indices], y_val[:, bp_indices], y_test[:, bp_indices]
    y_train_cc, y_val_cc, y_test_cc = y_train[:, cc_indices], y_val[:, cc_indices], y_test[:, cc_indices]

    # 2. Modality 1: Meta AI ESM-2 (320D)
    logger.info("Loading cached ESM-2 embeddings (320D)...")
    esm_train = np.load(processed_dir / "esm2_train.npy").astype(np.float32)
    esm_val = np.load(processed_dir / "esm2_val.npy").astype(np.float32)
    esm_test = np.load(processed_dir / "esm2_test.npy").astype(np.float32)

    # 3. Modality 2: 3-mer TF-IDF (1,500D)
    logger.info(f"Computing 3-mer TF-IDF representations ({kmer_features}D)...")
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=kmer_features, sublinear_tf=True)
    kmer_train = vec.fit_transform(train_df["sequence"]).toarray().astype(np.float32)
    kmer_val = vec.transform(val_df["sequence"]).toarray().astype(np.float32)
    kmer_test = vec.transform(test_df["sequence"]).toarray().astype(np.float32)
    logger.info(f"3-mer TF-IDF computed in {time.time() - t0:.2f}s (Shape: {kmer_train.shape})")

    # 4. Modality 3: Biophysical Descriptors (430D)
    logger.info("Loading and scaling 430D biophysical descriptors...")
    bio_train = np.load(processed_dir / "biophysical_train.npy")
    bio_val = np.load(processed_dir / "biophysical_val.npy")
    bio_test = np.load(processed_dir / "biophysical_test.npy")

    scaler = StandardScaler()
    bio_train_s = scaler.fit_transform(bio_train).astype(np.float32)
    bio_val_s = scaler.transform(bio_val).astype(np.float32)
    bio_test_s = scaler.transform(bio_test).astype(np.float32)

    # 5. Build Deep Tri-Modal Multi-Branch Fusion Architecture
    logger.info("Building Tri-Modal Multi-Branch Deep Network Architecture...")

    # Branch 1: ESM-2 Evolutionary Processing
    in_esm = Input(shape=(320,), name="esm_input")
    e = Dense(256, activation="relu", name="esm_dense1")(in_esm)
    e = BatchNormalization(name="esm_bn1")(e)
    e = Dropout(0.3, name="esm_drop1")(e)

    # Branch 2: 3-mer Sequence Motif Processing
    in_kmer = Input(shape=(kmer_features,), name="kmer_input")
    k = Dense(256, activation="relu", name="kmer_dense1")(in_kmer)
    k = BatchNormalization(name="kmer_bn1")(k)
    k = Dropout(0.3, name="kmer_drop1")(k)

    # Branch 3: Biophysical Descriptors Processing
    in_bio = Input(shape=(430,), name="bio_input")
    b = Dense(128, activation="relu", name="bio_dense1")(in_bio)
    b = BatchNormalization(name="bio_bn1")(b)
    b = Dropout(0.2, name="bio_drop1")(b)

    # Multi-Modal Cross-Fusion
    fused = Concatenate(name="fusion_concat")([e, k, b])
    f = Dense(512, activation="relu", name="fusion_dense1")(fused)
    f = BatchNormalization(name="fusion_bn1")(f)
    f = Dropout(0.35, name="fusion_drop1")(f)

    f = Dense(256, activation="relu", name="fusion_dense2")(f)
    f = BatchNormalization(name="fusion_bn2")(f)
    f = Dropout(0.25, name="fusion_drop2")(f)

    # Specialized Sub-Ontology Output Heads
    # Molecular Function (MF) Head
    mf_h = Dense(128, activation="relu", name="mf_dense")(f)
    mf_h = Dropout(0.2, name="mf_drop")(mf_h)
    out_mf = Dense(len(mf_indices), activation="sigmoid", name="out_mf")(mf_h)

    # Biological Process (BP) Head
    bp_h = Dense(128, activation="relu", name="bp_dense")(f)
    bp_h = Dropout(0.2, name="bp_drop")(bp_h)
    out_bp = Dense(len(bp_indices), activation="sigmoid", name="out_bp")(bp_head if 'bp_head' in locals() else bp_h)

    # Cellular Component (CC) Head
    cc_h = Dense(128, activation="relu", name="cc_dense")(f)
    cc_h = Dropout(0.2, name="cc_drop")(cc_h)
    out_cc = Dense(len(cc_indices), activation="sigmoid", name="out_cc")(cc_h)

    model = keras.Model(
        inputs=[in_esm, in_kmer, in_bio],
        outputs=[out_mf, out_bp, out_cc],
        name="ProteoPredict_TriModal_MultiBranch"
    )

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss={
            "out_mf": get_focal_loss(gamma=2.0, alpha=0.35),
            "out_bp": get_focal_loss(gamma=2.0, alpha=0.35),
            "out_cc": get_focal_loss(gamma=2.0, alpha=0.35),
        },
        loss_weights={
            "out_mf": 1.2,  # Molecular Function given high priority
            "out_bp": 1.0,
            "out_cc": 0.8,
        }
    )

    model.summary(print_fn=logger.info)

    # 6. Train Model
    checkpoint_path = models_dir / "trimodal_multibranch_best.h5"
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    train_inputs = [esm_train, kmer_train, bio_train_s]
    val_inputs = [esm_val, kmer_val, bio_val_s]
    test_inputs = [esm_test, kmer_test, bio_test_s]

    train_targets = {"out_mf": y_train_mf, "out_bp": y_train_bp, "out_cc": y_train_cc}
    val_targets = {"out_mf": y_val_mf, "out_bp": y_val_bp, "out_cc": y_val_cc}

    logger.info("Starting EXP-007 Tri-Modal Multi-Branch Training...")
    t_start = time.time()
    history = model.fit(
        x=train_inputs,
        y=train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_duration = time.time() - t_start
    logger.info(f"EXP-007 Training completed in {training_duration:.1f}s.")

    # 7. Validation Calibration (Per-Ontology Thresholds)
    logger.info("Calibrating per-ontology thresholds on validation set...")
    val_preds = model.predict(val_inputs, batch_size=batch_size)
    p_val_mf, p_val_bp, p_val_cc = val_preds[0], val_preds[1], val_preds[2]

    tau_mf, val_f1_mf = calibrate_branch_threshold(y_val_mf, p_val_mf)
    tau_bp, val_f1_bp = calibrate_branch_threshold(y_val_bp, p_val_bp)
    tau_cc, val_f1_cc = calibrate_branch_threshold(y_val_cc, p_val_cc)

    logger.info(f"Validation Calibrated Thresholds: MF={tau_mf:.3f} (F1={val_f1_mf:.4f}), BP={tau_bp:.3f} (F1={val_f1_bp:.4f}), CC={tau_cc:.3f} (F1={val_f1_cc:.4f})")

    # 8. Unseen Holdout Test Set Evaluation
    logger.info("Evaluating on strictly non-homologous holdout test set...")
    test_preds = model.predict(test_inputs, batch_size=batch_size)
    p_test_mf, p_test_bp, p_test_cc = test_preds[0], test_preds[1], test_preds[2]

    y_test_bin_mf = (p_test_mf >= tau_mf).astype(int)
    y_test_bin_bp = (p_test_bp >= tau_bp).astype(int)
    y_test_bin_cc = (p_test_cc >= tau_cc).astype(int)

    test_f1_mf = float(f1_score(y_test_mf, y_test_bin_mf, average="micro", zero_division=0))
    test_f1_bp = float(f1_score(y_test_bp, y_test_bin_bp, average="micro", zero_division=0))
    test_f1_cc = float(f1_score(y_test_cc, y_test_bin_cc, average="micro", zero_division=0))

    # Reconstruct full 75-class test predictions
    y_test_pred_full = np.zeros_like(y_test)
    y_test_pred_full[:, mf_indices] = y_test_bin_mf
    y_test_pred_full[:, bp_indices] = y_test_bin_bp
    y_test_pred_full[:, cc_indices] = y_test_bin_cc

    test_micro = float(f1_score(y_test, y_test_pred_full, average="micro", zero_division=0))
    test_macro = float(f1_score(y_test, y_test_pred_full, average="macro", zero_division=0))
    test_prec = float(precision_score(y_test, y_test_pred_full, average="micro", zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred_full, average="micro", zero_division=0))

    # Also compute overall validation Fmax
    val_pred_full = np.zeros_like(y_val)
    val_pred_full[:, mf_indices] = (p_val_mf >= tau_mf).astype(int)
    val_pred_full[:, bp_indices] = (p_val_bp >= tau_bp).astype(int)
    val_pred_full[:, cc_indices] = (p_val_cc >= tau_cc).astype(int)
    val_micro = float(f1_score(y_val, val_pred_full, average="micro", zero_division=0))

    metrics = {
        "model": "Tri-Modal Multi-Branch Deep Network (EXP-007)",
        "calibrated_thresholds": {
            "MF": tau_mf,
            "BP": tau_bp,
            "CC": tau_cc
        },
        "validation_fmax": val_micro,
        "validation_f1_mf": val_f1_mf,
        "validation_f1_bp": val_f1_bp,
        "validation_f1_cc": val_f1_cc,
        "test_micro_f1": test_micro,
        "test_macro_f1": test_macro,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1_mf": test_f1_mf,
        "test_f1_bp": test_f1_bp,
        "test_f1_cc": test_f1_cc,
        "training_time_s": round(training_duration, 1),
        "kmer_features": kmer_features
    }

    metrics_path = results_dir / "trimodal_multibranch_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.info("FINAL BENCHMARK: TRI-MODAL MULTI-BRANCH DEEP NETWORK (EXP-007)")
    logger.info("=" * 65)
    logger.info(f"Thresholds (tau*):     MF={tau_mf:.3f}, BP={tau_bp:.3f}, CC={tau_cc:.3f}")
    logger.info(f"Validation F_max:      {val_micro:.4f}")
    logger.info(f"Test Precision:        {test_prec:.4f}")
    logger.info(f"Test Recall:           {test_rec:.4f}")
    logger.info(f"Test Micro-F1 (F_max): {test_micro:.4f}")
    logger.info(f"Test Macro-F1:         {test_macro:.4f}")
    logger.info(f"F1 (Molecular Func):   {test_f1_mf:.4f}")
    logger.info(f"F1 (Biological Proc):  {test_f1_bp:.4f}")
    logger.info(f"F1 (Cellular Comp):    {test_f1_cc:.4f}")
    logger.info("=" * 65)

    return metrics


if __name__ == "__main__":
    train_trimodal_multibranch()
