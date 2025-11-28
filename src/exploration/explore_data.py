"""
✅ Data Exploration Script for ProteoPredict
--------------------------------------------
This script performs exploratory data analysis (EDA)
on processed protein sequence datasets before model training.
It helps visualize sequence lengths, label distribution,
and get a basic overview of the dataset.

Input: data/processed/train_data.npz
Output: reports/plots/*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_PATH = "data/processed/train_data.npz"
PLOT_DIR = "reports/plots"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Load dataset
print("🔍 Loading processed dataset...")
data = np.load(DATA_PATH, allow_pickle=True)
X = data["X"]
y = data["y"]

print(f"✅ Dataset loaded successfully!")
print(f"🧬 X shape: {X.shape}")
print(f"🏷️  y shape: {y.shape}")

# ---------------- Sequence Length Analysis ----------------
seq_lengths = [len(seq) for seq in X]
plt.figure(figsize=(8, 5))
sns.histplot(seq_lengths, bins=30, kde=True, color="skyblue")
plt.title("Distribution of Protein Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "sequence_length_distribution.png"))
plt.close()

print("📊 Saved: sequence_length_distribution.png")

# ---------------- Label Distribution ----------------
label_sums = np.sum(y, axis=0)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(range(len(label_sums))), y=label_sums, color="teal")
plt.title("Label Frequency Distribution")
plt.xlabel("Label Index")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "label_distribution.png"))
plt.close()

print("📊 Saved: label_distribution.png")

# ---------------- Summary Statistics ----------------
print("\n📈 Dataset Summary:")
print(f"Average Sequence Length: {np.mean(seq_lengths):.2f}")
print(f"Max Sequence Length: {np.max(seq_lengths)}")
print(f"Min Sequence Length: {np.min(seq_lengths)}")
print(f"Total Labels: {y.shape[1]}")
print(f"Average Labels per Sample: {np.mean(np.sum(y, axis=1)):.2f}")

print("\n✅ Data exploration complete! Plots saved in 'reports/plots/' directory.")
