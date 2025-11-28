"""
ProteoPredict - Model Training Module
Builds and trains deep learning models for protein function prediction
Includes Custom Weighted Binary Cross-Entropy Loss (WBCEL) for imbalance.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D,
    Dense, Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from pathlib import Path

# ----------------------- CUSTOM LOSS FUNCTION -----------------------

def weighted_binary_crossentropy(pos_weight_tensor):
    """
    Returns a custom weighted binary crossentropy function.
    pos_weight_tensor: 1D tensor/array of shape (num_classes,)
    The returned loss has shape (batch,) (mean over classes).
    """
    # Make sure pos_weight_tensor is a tf tensor of dtype float32
    pos_w = tf.cast(pos_weight_tensor, tf.float32)

    def loss(y_true, y_pred):
        # Cast to float32 to avoid dtype mismatches
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # Compute elementwise binary crossentropy (per-sample, per-class)
        # Use K.binary_crossentropy which returns shape (batch, num_classes)
        bce = K.binary_crossentropy(y_true_f, y_pred_f)

        # Weighted BCE: pos_w * y_true + (1 - y_true)
        # pos_w shape (num_classes,), bce shape (batch, num_classes) -> broadcasting
        weight_factor = pos_w * y_true_f + (1.0 - y_true_f)
        weighted_bce = weight_factor * bce

        # Mean across classes for each sample
        return K.mean(weighted_bce, axis=-1)

    # Add an attribute for easier custom_objects handling later
    loss._is_weighted_bce = True
    return loss


# --- DATA LOADING ---

def load_data(data_dir):
    """Load preprocessed data"""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")

    data_dir = Path(data_dir)
    train_data = np.load(data_dir / "train_data.npz")
    val_data = np.load(data_dir / "val_data.npz")

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']

    with open(data_dir / "preprocess_config.json", 'r') as f:
        config = json.load(f)

    vocab_size = int(config['vocab_size'])
    max_length = int(config['max_length'])

    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"✓ Vocab size: {vocab_size}, Max length: {max_length}")
    print(f"✓ GO classes: {y_train.shape[1]}")

    max_train_idx, max_val_idx = int(X_train.max()), int(X_val.max())
    if max_train_idx >= vocab_size or max_val_idx >= vocab_size:
        vocab_size = int(max(max_train_idx, max_val_idx)) + 10
        print(f"⚠️ Adjusted vocab_size to {vocab_size}")

    return X_train, X_val, y_train, y_val, vocab_size, max_length


# ----------------------- MODEL DEFINITIONS -----------------------

def get_loss_function(pos_weight):
    if pos_weight is not None:
        print("Using Custom Weighted Binary Cross-Entropy Loss.")
        return weighted_binary_crossentropy(pos_weight)
    else:
        print("Using standard Binary Cross-Entropy Loss.")
        return 'binary_crossentropy'


def build_baseline_model(vocab_size, num_classes, max_length, embedding_dim=64, pos_weight=None):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss=get_loss_function(pos_weight),
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


def build_lstm_model(vocab_size, num_classes, max_length, embedding_dim=128, pos_weight=None):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=get_loss_function(pos_weight),
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


def build_cnn_model(vocab_size, num_classes, max_length, embedding_dim=128, pos_weight=None):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(256, 5, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=get_loss_function(pos_weight),
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


def build_hybrid_model(vocab_size, num_classes, max_length, embedding_dim=128, pos_weight=None):
    """Hybrid CNN + LSTM Model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(256, 5, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=get_loss_function(pos_weight),
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


# ----------------------- TRAINING AND EVALUATION -----------------------

def train_model(model, X_train, y_train, X_val, y_val, output_dir, epochs=20, batch_size=32):
    print(f"\n{'='*70}")
    print("TRAINING MODEL")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(output_dir / 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            save_weights_only=False,
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=str(output_dir / 'logs'), histogram_freq=1)
    ]

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(str(output_dir / 'final_model.h5'))
    print(f"\n✓ Final model saved to: {output_dir / 'final_model.h5'}")

    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        hist = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(hist, f, indent=2)
    print("✓ Training history saved.")

    return history


def find_optimal_threshold(model, X_val, y_val, num_steps=100):
    """
    Finds the optimal threshold (micro F1) over a wide range.
    """
    y_pred_proba = model.predict(X_val, verbose=0)
    thresholds = np.linspace(0.01, 0.9, num_steps)
    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        y_pred_bin = (y_pred_proba > t).astype(int)
        f1_micro = f1_score(y_val, y_pred_bin, average='micro', zero_division=0)
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_threshold = t

    return best_threshold, best_f1


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model using the optimal threshold and reports all metrics.
    """
    y_pred_proba = model.predict(X_val, verbose=0)

    optimal_threshold, micro_f1_optimal = find_optimal_threshold(model, X_val, y_val)

    y_pred_bin = (y_pred_proba > optimal_threshold).astype(int)

    precision_w = precision_score(y_val, y_pred_bin, average='weighted', zero_division=0)
    recall_w = recall_score(y_val, y_pred_bin, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_val, y_pred_bin, average='weighted', zero_division=0)

    print(f"\nValidation Results (using optimal threshold: {optimal_threshold:.4f}):")
    print(f"  Precision (Weighted): {precision_w:.4f}")
    print(f"  Recall (Weighted): {recall_w:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"  F1-Score (Micro): {micro_f1_optimal:.4f} (Key metric for GO prediction)")

    # Optionally return the raw predictions so notebook can plot curves
    return {
        'precision': precision_w,
        'recall': recall_w,
        'f1_score_weighted': f1_weighted,
        'f1_score_micro': micro_f1_optimal,
        'threshold': optimal_threshold,
        'y_pred_proba': y_pred_proba.tolist()  # convert to list for json-safety if needed
    }


# ----------------------- MAIN FUNCTION -----------------------

def main():
    parser = argparse.ArgumentParser(description='Train ProteoPredict model')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='models/weighted_run')
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['baseline', 'cnn', 'lstm', 'hybrid'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=128)
    args = parser.parse_args()

    print("\n" + "🧬 " * 20)
    print("PROTEOPREDICT - MODEL TRAINING STARTED (with Weighted Loss)")
    print("🧬 " * 20 + "\n")

    X_train, X_val, y_train, y_val, vocab_size, max_length = load_data(args.data_dir)
    num_classes = int(y_train.shape[1])

    print("\n================ DEBUG INFO ================")
    print("y_train shape:", y_train.shape)
    print("Example label (first row):", y_train[0][:10])
    print("Unique label values (sample):", np.unique(y_train)[:10])

    if y_train.ndim == 1 or y_train.shape[1] == 1:
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))
    else:
        label_sum = y_train.sum(axis=0)
        print("Positives per class (first 10):", label_sum[:10])
        print("Average positives per class:", np.mean(label_sum))
    print("============================================\n")

    # ------------ Positive weight calculation ------------
    total_samples = float(y_train.shape[0])
    positive_counts = np.array(y_train.sum(axis=0), dtype=np.float32)
    # Avoid division by zero
    pos_weight = (total_samples - positive_counts) / (positive_counts + 1.0)
    # Convert to tensor
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    print(f"✓ Calculated custom positive weights (first 10): {pos_weight_tensor.numpy()[:10]}")
    # -----------------------------------------------------

    model_map = {
        'baseline': build_baseline_model,
        'cnn': build_cnn_model,
        'lstm': build_lstm_model,
        'hybrid': build_hybrid_model
    }

    model = model_map[args.model_type](
        vocab_size,
        num_classes,
        max_length,
        args.embedding_dim,
        pos_weight=pos_weight_tensor
    )

    history = train_model(model, X_train, y_train, X_val, y_val,
                          args.output_dir, args.epochs, args.batch_size)

    metrics = evaluate_model(model, X_val, y_val)

    optimal_threshold = metrics['threshold']
    preds = model.predict(X_val[:100])
    print("\n===== DEBUG PREDICTIONS =====")
    print(f"Unique predicted labels (using threshold {optimal_threshold:.4f}):",
          np.unique((preds > optimal_threshold).astype(int)))
    print("==============================\n")

    # Save metrics (without huge arrays) and also save predictions separately if desired
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove large y_pred_proba before saving summary
    saved_metrics = {k: v for k, v in metrics.items() if k != 'y_pred_proba'}
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(saved_metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {out_dir / 'metrics.json'}")

    # Optionally save predictions to a npz for plotting later
    y_pred_proba = np.array(metrics.get('y_pred_proba')) if 'y_pred_proba' in metrics else model.predict(X_val, verbose=0)
    np.savez_compressed(out_dir / 'predictions.npz', y_true=y_val, y_pred_proba=y_pred_proba)
    print(f"✓ Predictions saved to: {out_dir / 'predictions.npz'}")

    print(f"\n✅ TRAINING COMPLETE! Best Micro F1-Score: {metrics['f1_score_micro']:.4f}")


if __name__ == "__main__":
    main()
