"""
ProteoPredict - SOTA Inference and Prediction Engine (EXP-007)
Generates authentic functional Gene Ontology predictions from amino acid sequences
using the Tri-Modal Multi-Branch Deep Fusion Network (ESM-2 + 3-mer TF-IDF + Biophysical)
under strict validation-calibrated sub-ontology thresholds.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import pandas as pd
import tensorflow as tf

from src.proteopredict.features.biophysical import batch_extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.Predict")

# Global singleton cache for inference artifacts to ensure fast sub-second predictions
_GLOBAL_ARTIFACTS = {
    "model": None,
    "esm_model": None,
    "batch_converter": None,
    "vectorizer": None,
    "scaler": None,
    "metadata": None,
    "go_names": None,
}


def load_inference_engine(
    model_path: str = "models/trimodal_multibranch_best.h5",
    meta_path: str = "models/trimodal_metadata.json",
    vectorizer_path: str = "models/kmer_vectorizer_1500.pkl",
    scaler_path: str = "models/biophysical_scaler.pkl",
    go_names_path: str = "data/processed/go_names.json",
):
    """Loads and caches all Tri-Modal models, PLM weights, and scalers in memory."""
    global _GLOBAL_ARTIFACTS

    if _GLOBAL_ARTIFACTS["model"] is not None:
        return _GLOBAL_ARTIFACTS

    logger.info("Initializing ProteoPredict Tri-Modal Inference Engine...")

    # 1. Load Pre-trained ESM-2
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    _GLOBAL_ARTIFACTS["esm_model"] = esm_model
    _GLOBAL_ARTIFACTS["batch_converter"] = batch_converter

    # 2. Load Vectorizer & Scaler
    with open(vectorizer_path, "rb") as f:
        _GLOBAL_ARTIFACTS["vectorizer"] = pickle.load(f)

    with open(scaler_path, "rb") as f:
        _GLOBAL_ARTIFACTS["scaler"] = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        _GLOBAL_ARTIFACTS["metadata"] = json.load(f)

    if Path(go_names_path).exists():
        with open(go_names_path, "r", encoding="utf-8") as f:
            _GLOBAL_ARTIFACTS["go_names"] = json.load(f)
    else:
        _GLOBAL_ARTIFACTS["go_names"] = {}

    # 3. Load Tri-Modal Multi-Branch Deep Network
    _GLOBAL_ARTIFACTS["model"] = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Tri-Modal Inference Engine loaded successfully!")
    return _GLOBAL_ARTIFACTS


def extract_tri_modal_features(sequence: str, artifacts: dict) -> tuple:
    """Extracts 320D ESM-2, 1,500D 3-mer TF-IDF, and 430D Biophysical features on the fly."""
    clean_seq = str(sequence).strip().upper()

    # 1. ESM-2 Evolutionary Embedding (320D)
    esm_model = artifacts["esm_model"]
    batch_converter = artifacts["batch_converter"]
    seq_trunc = clean_seq[:1024]
    _, _, tokens = batch_converter([("query", seq_trunc)])
    with torch.no_grad():
        res = esm_model(tokens, repr_layers=[6], return_contacts=False)
        esm_feat = res["representations"][6][0, 1 : len(seq_trunc) + 1].mean(0).numpy().astype(np.float32).reshape(1, -1)

    # 2. 3-mer TF-IDF Sequence Motif Vector (1,500D)
    vec = artifacts["vectorizer"]
    kmer_feat = vec.transform([clean_seq]).toarray().astype(np.float32)

    # 3. Biophysical Profile (430D)
    raw_bio = batch_extract_features([clean_seq])
    scaler = artifacts["scaler"]
    bio_feat = scaler.transform(raw_bio).astype(np.float32)

    return esm_feat, kmer_feat, bio_feat


def predict_go_terms(
    sequence: str,
    model_path: str = "models/trimodal_multibranch_best.h5",
    data_dir: str = "data/processed",
    return_results: bool = True,
    custom_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Predict Gene Ontology terms for an input protein sequence using the
    state-of-the-art Tri-Modal Multi-Branch Deep Fusion Network (EXP-007).
    """
    if not sequence or len(str(sequence).strip()) < 10:
        raise ValueError("Valid amino acid sequence of at least 10 residues required.")

    clean_seq = "".join([c for c in str(sequence).strip().upper() if c.isalpha()])

    # Load engine
    artifacts = load_inference_engine(model_path=model_path)
    model = artifacts["model"]
    meta = artifacts["metadata"]
    go_names = artifacts["go_names"]

    # Extract all three modalities
    esm_feat, kmer_feat, bio_feat = extract_tri_modal_features(clean_seq, artifacts)

    # Multi-Branch Prediction
    p_mf, p_bp, p_cc = model.predict([esm_feat, kmer_feat, bio_feat], verbose=0)

    # Default calibrated thresholds
    tau_mf = meta["thresholds"]["MF"]
    tau_bp = meta["thresholds"]["BP"]
    tau_cc = meta["thresholds"]["CC"]

    if custom_threshold is not None:
        tau_mf = float(custom_threshold)
        tau_bp = float(custom_threshold)
        tau_cc = float(custom_threshold)

    results = []
    category_map = {"MF": "Molecular Function", "BP": "Biological Process", "CC": "Cellular Component"}

    # Process MF predictions
    for i, term in enumerate(meta["mf_classes"]):
        prob = float(p_mf[0, i])
        if prob >= tau_mf:
            name = go_names.get(term, term)
            results.append({
                "GO_ID": term,
                "Predicted_Function": name,
                "Confidence_Score": round(prob, 4),
                "Category_Code": "MF",
                "Category": category_map["MF"],
                "Description": name,
                "Probability": round(prob, 4),
                "Confidence (%)": f"{round(prob * 100, 2)}%",
                "Calibrated_Threshold": round(tau_mf, 3)
            })

    # Process BP predictions
    for i, term in enumerate(meta["bp_classes"]):
        prob = float(p_bp[0, i])
        if prob >= tau_bp:
            name = go_names.get(term, term)
            results.append({
                "GO_ID": term,
                "Predicted_Function": name,
                "Confidence_Score": round(prob, 4),
                "Category_Code": "BP",
                "Category": category_map["BP"],
                "Description": name,
                "Probability": round(prob, 4),
                "Confidence (%)": f"{round(prob * 100, 2)}%",
                "Calibrated_Threshold": round(tau_bp, 3)
            })

    # Process CC predictions
    for i, term in enumerate(meta["cc_classes"]):
        prob = float(p_cc[0, i])
        if prob >= tau_cc:
            name = go_names.get(term, term)
            results.append({
                "GO_ID": term,
                "Predicted_Function": name,
                "Confidence_Score": round(prob, 4),
                "Category_Code": "CC",
                "Category": category_map["CC"],
                "Description": name,
                "Probability": round(prob, 4),
                "Confidence (%)": f"{round(prob * 100, 2)}%",
                "Calibrated_Threshold": round(tau_cc, 3)
            })

    results.sort(key=lambda x: x["Confidence_Score"], reverse=True)

    logger.info(f"Predicted {len(results)} functions for sequence (length: {len(clean_seq)} aa)")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict Gene Ontology functions for a protein sequence.")
    parser.add_argument("--sequence", type=str, required=True, help="Input amino acid sequence.")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (optional).")
    args = parser.parse_args()

    results = predict_go_terms(sequence=args.sequence, custom_threshold=args.threshold)
    print("\n" + "=" * 65)
    print("PROTEOPREDICT TRI-MODAL PREDICTION RESULTS (EXP-007)")
    print(f"Sequence Length: {len(args.sequence.strip())} aa")
    print(f"Total Predictions: {len(results)}")
    print("=" * 65)
    if results:
        df = pd.DataFrame(results)
        print(df[["GO_ID", "Category_Code", "Predicted_Function", "Confidence_Score", "Calibrated_Threshold"]].to_string(index=False))
    else:
        print("No terms exceeded the calibrated confidence thresholds.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()