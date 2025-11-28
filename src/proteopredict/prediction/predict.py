# src/proteopredict/prediction/predict.py (FINAL, CORRECTED VERSION)

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import json

# --- Helper Function for Preprocessing ---

def preprocess_sequence(sequence, char_to_index, max_length):
    """Converts protein sequence to padded token indices."""
    
    if not isinstance(char_to_index, dict):
        raise TypeError("char_to_index must be a dictionary for mapping tokens.")

    token_indices = [char_to_index.get(char, 0) for char in sequence]
    
    if len(token_indices) > max_length:
        token_indices = token_indices[:max_length]
    elif len(token_indices) < max_length:
        padding = [0] * (max_length - len(token_indices))
        token_indices.extend(padding)
        
    return np.array([token_indices])


# --- Main Prediction Logic ---

def predict_go_terms(model_path, data_dir, sequence, return_results=False, custom_threshold=None):
    print("\n" + "🧬 " * 15)
    print("PROTEOPREDICT - FINAL PREDICTION STARTED")
    print("🧬 " * 15 + "\n")
    
    model_path = Path(model_path)
    data_dir = Path(data_dir)

    # 1. Load Configs
    with open(data_dir / "preprocess_config.json", 'r') as f:
        config = json.load(f)
    
    # 🌟 FIX 1: Use the ACTUAL key found for amino acid mapping
    char_to_index = config['aa_to_int']
    max_length = config['max_length']
    
    # 🌟 FIX 2: Load the GO mapping from go_info.json and convert the list to a dictionary
    try:
        with open(data_dir / "go_info.json", 'r') as f:
            go_info = json.load(f)
            # The list of classes acts as the index-to-GO_ID map (Index 0 -> GO:0000166, etc.)
            go_id_list = go_info['classes'] 
            # Convert list to dictionary {0: 'GO:0000166', 1: 'GO:0003677', ...}
            index_to_label = {str(i): go_id for i, go_id in enumerate(go_id_list)}
        print(f"✓ Loaded GO index mapping from go_info.json. Total classes: {len(go_id_list)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"GO info file not found at {data_dir / 'go_info.json'}")


    # Load GO Names for human readability (Requires go_names.json to be created by you)
    try:
        with open(data_dir / "go_names.json", 'r') as f:
            go_name_map = json.load(f)
        print("✓ Loaded GO term descriptions (for presentation).")
    except FileNotFoundError:
        print("⚠️ WARNING: 'go_names.json' not found. Outputting GO IDs only.")
        go_name_map = {} 

    
    # Load optimal threshold from metrics.json (saved from training)
    try:
        with open(model_path.parent / "metrics.json", 'r') as f:
            metrics = json.load(f)
            # Use custom threshold from Streamlit, otherwise fall back to optimal_threshold from metrics.json
        optimal_threshold = metrics.get('threshold', 0.15) 
        if custom_threshold is not None:
            optimal_threshold = custom_threshold
    except:
        optimal_threshold = 0.15 
        print(f"⚠️ WARNING: Could not load optimal threshold from metrics.json. Using default: {optimal_threshold}")

    # 2. Preprocess Sequence
    X_new = preprocess_sequence(sequence, char_to_index, max_length)
    print(f"Preprocessed sequence shape: {X_new.shape}")

    # 3. Load Model
    print(f"Loading model from: {model_path}")
    
    # Robust Model Loading (Handles custom loss)
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Attempting to load model with custom_objects due to previous loss function...")
    
        from src.training.train import weighted_binary_crossentropy
        num_classes = len(index_to_label)
        dummy_pos_weight = tf.constant(np.ones(num_classes), dtype=tf.float32)

        model = load_model(
            model_path, 
            custom_objects={'loss': weighted_binary_crossentropy(dummy_pos_weight)}
        )
        print("✓ Model loaded successfully with custom loss definition.")


    # 4. Predict and Output
    y_pred_proba = model.predict(X_new, verbose=0)[0]
    
    # 5. Apply Threshold
    y_pred_bin = (y_pred_proba > optimal_threshold).astype(int)
    
    # --- Preparation for Terminal Output & Streamlit Return ---
    
    # Initialize the list to be returned to Streamlit
    final_results = []
    predicted_count = 0
    
    # 6. Loop through all classes and collect/print results
    
    print(f"\n{'='*70}")
    print(f"FINAL PREDICTION FOR UNKNOWN SEQUENCE: {sequence[:30]}...")
    print(f"Optimal Threshold Used: {optimal_threshold:.4f} (from metrics.json)")
    print(f"{'-'*70}")
    
    for i, is_active in enumerate(y_pred_bin):
        # Check if prediction is above the threshold
        if is_active == 1:
            predicted_count += 1
            # Index must be converted to string because JSON keys are strings
            go_id = index_to_label[str(i)] 
            confidence = y_pred_proba[i]
            
            # Use GO Name if available
            go_name = go_name_map.get(go_id, go_id)
            
            # 1. Print to Terminal (like before)
            print(f"[{go_id}] → {go_name} (Confidence: {confidence:.4f})")

            # 2. Collect for Streamlit (CRITICAL FIX: Collecting the data here)
            final_results.append([go_id, go_name, confidence]) # Store confidence as float!
            
    if predicted_count == 0:
        print("No GO terms predicted above the optimal threshold.")
        
    print(f"Total predicted GO terms: {predicted_count}")
    print(f"{'='*70}")
    
    # --- Streamlit Return Logic ---
    if return_results:
        return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict GO terms for a new protein sequence.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the final Keras model (.h5 file).')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the preprocessing config file.')
    parser.add_argument('--sequence', type=str, 
                        help='The new protein sequence to predict (e.g., MKTLLIALA...).')
    args = parser.parse_args()
    
    if args.sequence:
        predict_go_terms(args.model_path, args.data_dir, args.sequence)
    else:
        print("Please provide a protein sequence using the --sequence argument.")