import json
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

processed_dir = Path("data/processed")
models_dir = Path("models")
results_dir = Path("results")

# Load data
val_npz = np.load(processed_dir / "val_data.npz")
test_npz = np.load(processed_dir / "test_data.npz")

X_val_seq, y_val = val_npz["X"], val_npz["y"]
X_test_seq, y_test = test_npz["X"], test_npz["y"]

bio_train = np.load(processed_dir / "biophysical_train.npy")
bio_val = np.load(processed_dir / "biophysical_val.npy")
bio_test = np.load(processed_dir / "biophysical_test.npy")

scaler = StandardScaler()
scaler.fit(bio_train)
bio_val_scaled = scaler.transform(bio_val)
bio_test_scaled = scaler.transform(bio_test)

with open(processed_dir / "go_info.json", "r") as f:
    go_info = json.load(f)
classes = go_info["classes"]
go_metadata = go_info.get("term_metadata", {})

print("Loading saved best model: models/resnet_bilstm_attn_best.h5...")
model = load_model("models/resnet_bilstm_attn_best.h5", compile=False)

print("Predicting on validation set...")
val_probs = model.predict([X_val_seq, bio_val_scaled], batch_size=64, verbose=0)

# Calibrate tau* on validation
best_f1, best_tau = 0.0, 0.25
for t in np.linspace(0.05, 0.70, 60):
    f1 = f1_score(y_val, (val_probs >= t).astype(int), average="micro", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_tau = float(t)

print(f"Validation Calibrated tau*: {best_tau:.3f} (Val F1: {best_f1:.4f})")

print("Predicting on unseen test set...")
test_probs = model.predict([X_test_seq, bio_test_scaled], batch_size=64, verbose=0)
y_pred_bin = (test_probs >= best_tau).astype(int)

prec = float(precision_score(y_test, y_pred_bin, average="micro", zero_division=0))
rec = float(recall_score(y_test, y_pred_bin, average="micro", zero_division=0))
f1_micro = float(f1_score(y_test, y_pred_bin, average="micro", zero_division=0))
f1_macro = float(f1_score(y_test, y_pred_bin, average="macro", zero_division=0))

mf_idx = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "MF"]
bp_idx = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "BP"]
cc_idx = [i for i, t in enumerate(classes) if go_metadata.get(t, {}).get("category") == "CC"]

f1_mf = float(f1_score(y_test[:, mf_idx], y_pred_bin[:, mf_idx], average="micro", zero_division=0))
f1_bp = float(f1_score(y_test[:, bp_idx], y_pred_bin[:, bp_idx], average="micro", zero_division=0))
f1_cc = float(f1_score(y_test[:, cc_idx], y_pred_bin[:, cc_idx], average="micro", zero_division=0))

results = {
    "model": "Deep ResNet-BiLSTM-Attention Hybrid (EXP-005)",
    "calibrated_threshold": best_tau,
    "validation_fmax": best_f1,
    "precision_micro": prec,
    "recall_micro": rec,
    "f1_micro": f1_micro,
    "f1_macro": f1_macro,
    "f1_mf": f1_mf,
    "f1_bp": f1_bp,
    "f1_cc": f1_cc
}

with open(results_dir / "resnet_bilstm_attn_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 50)
print(f"Test Micro-F1 (F_max): {f1_micro:.4f}")
print(f"Test Macro-F1:         {f1_macro:.4f}")
print(f"MF F1:                 {f1_mf:.4f}")
print(f"BP F1:                 {f1_bp:.4f}")
print(f"CC F1:                 {f1_cc:.4f}")
print("=" * 50)
