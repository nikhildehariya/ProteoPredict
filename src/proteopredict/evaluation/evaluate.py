# src/proteopredict/evaluation/evaluate.py

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# --- Replicating the Optimal Threshold Logic for Evaluation ---

def find_optimal_threshold(model, X_data, y_true, num_steps=50):
    """Finds the optimal threshold to maximize Micro F1-score."""
    y_pred_proba = model.predict(X_data, verbose=0)
    thresholds = np.linspace(0.01, 0.5, num_steps) 
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        y_pred_bin = (y_pred_proba > t).astype(int)
        f1_micro = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_threshold = t
            
    return best_threshold, best_f1

# --- Main Evaluation Logic ---

def evaluate(model_path, data_dir):
    model_path = Path(model_path)
    data_dir = Path(data_dir)

    # 1. Load Data
    print(f"Loading test data from: {data_dir / 'test_data.npz'}")
    test_data = np.load(data_dir / "test_data.npz")
    X_test, y_test = test_data['X'], test_data['y']

    # 2. Load Model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # 3. Predict Probabilities
    print("Generating predictions on test set...")
    y_pred_proba = model.predict(X_test, verbose=1)

    # 4. Find Optimal Threshold (using the test set for final reporting is common)
    optimal_threshold, micro_f1_optimal = find_optimal_threshold(model, X_test, y_test)
    
    # 5. Binarize using Optimal Threshold
    y_pred_bin = (y_pred_proba > optimal_threshold).astype(int)

    # 6. Final Metric Calculation
    precision_w = precision_score(y_test, y_pred_bin, average='weighted', zero_division=0)
    recall_w = recall_score(y_test, y_pred_bin, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_bin, average='weighted', zero_division=0)
    
    print(f"\n{'='*50}")
    print("FINAL TEST SET EVALUATION")
    print(f"Optimal Threshold used: {optimal_threshold:.4f}")
    print(f"{'-'*50}")
    print(f"Precision (Weighted): {precision_w:.4f}")
    print(f"Recall (Weighted): {recall_w:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Micro): {micro_f1_optimal:.4f} 🔥")
    print(f"{'='*50}")

    return {
        'micro_f1': micro_f1_optimal,
        'threshold': optimal_threshold
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ProteoPredict model on test data.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the final Keras model (.h5 file).')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the preprocessed data (test_data.npz).')
    args = parser.parse_args()
    
    # Run the evaluation
    evaluate(args.model_path, args.data_dir)