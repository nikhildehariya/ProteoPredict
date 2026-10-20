import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
models_dir = Path("models")

# 1. Fit & save 3-mer TF-IDF Vectorizer
print("Fitting 3-mer TF-IDF vectorizer (1,500D)...")
train_df = pd.read_csv(raw_dir / "train.csv")
vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=1500, sublinear_tf=True)
vec.fit(train_df["sequence"])

vec_path = models_dir / "kmer_vectorizer_1500.pkl"
with open(vec_path, "wb") as f:
    pickle.dump(vec, f)
print(f"Saved {vec_path} ({len(vec.vocabulary_)} features)")

# 2. Fit & save Biophysical StandardScaler
print("Fitting Biophysical StandardScaler (430D)...")
bio_train = np.load(processed_dir / "biophysical_train.npy")
scaler = StandardScaler()
scaler.fit(bio_train)

scaler_path = models_dir / "biophysical_scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Saved {scaler_path}")

# 3. Save Tri-Modal Metadata for Inference
with open(processed_dir / "go_info.json", "r") as f:
    go_info = json.load(f)
classes = go_info["classes"]
term_metadata = go_info.get("term_metadata", {})

with open("results/trimodal_multibranch_metrics.json", "r") as f:
    exp_metrics = json.load(f)

mf_indices = [i for i, t in enumerate(classes) if term_metadata.get(t, {}).get("category") == "MF"]
bp_indices = [i for i, t in enumerate(classes) if term_metadata.get(t, {}).get("category") == "BP"]
cc_indices = [i for i, t in enumerate(classes) if term_metadata.get(t, {}).get("category") == "CC"]

metadata = {
    "model_name": "Tri-Modal Multi-Branch Deep Network (EXP-007)",
    "num_classes": len(classes),
    "classes": classes,
    "mf_indices": mf_indices,
    "bp_indices": bp_indices,
    "cc_indices": cc_indices,
    "mf_classes": [classes[i] for i in mf_indices],
    "bp_classes": [classes[i] for i in bp_indices],
    "cc_classes": [classes[i] for i in cc_indices],
    "thresholds": exp_metrics["calibrated_thresholds"],
    "metrics": {
        "test_micro_f1": exp_metrics["test_micro_f1"],
        "test_macro_f1": exp_metrics["test_macro_f1"],
        "test_f1_mf": exp_metrics["test_f1_mf"],
        "test_f1_bp": exp_metrics["test_f1_bp"],
        "test_f1_cc": exp_metrics["test_f1_cc"],
        "test_precision": exp_metrics["test_precision"],
        "test_recall": exp_metrics["test_recall"],
    }
}

meta_path = models_dir / "trimodal_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved {meta_path}")
