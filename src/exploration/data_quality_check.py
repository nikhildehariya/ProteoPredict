"""
Data Quality Check for ProteoPredict
- Read-only checks on processed .npz files
- Saves human-readable JSON report and plots in reports/
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPORT_DIR = Path("reports")
PLOTS_DIR = REPORT_DIR / "plots"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    acc = d.get("accessions", None)
    return X, y, acc

def safe_stats(X):
    # X is integer-encoded array (N, L)
    seq_lengths = (X != 0).sum(axis=1)  # count non-pad tokens assuming 0 is PAD
    return {
        "n_samples": X.shape[0],
        "max_token": int(X.max()),
        "min_token": int(X.min()),
        "seq_len_mean": float(seq_lengths.mean()),
        "seq_len_median": float(np.median(seq_lengths)),
        "seq_len_min": int(seq_lengths.min()),
        "seq_len_max": int(seq_lengths.max()),
        "padding_fraction_mean": float((X == 0).mean())
    }

def label_stats(y):
    n_samples, n_classes = y.shape
    labels_per_sample = y.sum(axis=1)
    per_label_counts = y.sum(axis=0)
    zero_label_samples = int((labels_per_sample == 0).sum())
    sparsity = float(1.0 - (y.mean()))
    top_labels = np.argsort(per_label_counts)[::-1][:10].tolist()
    bottom_labels = np.argsort(per_label_counts)[:10].tolist()
    return {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "avg_labels_per_sample": float(labels_per_sample.mean()),
        "median_labels_per_sample": float(np.median(labels_per_sample)),
        "min_labels_per_sample": int(labels_per_sample.min()),
        "max_labels_per_sample": int(labels_per_sample.max()),
        "samples_with_zero_labels": zero_label_samples,
        "sparsity": sparsity,
        "top_10_label_indices": top_labels,
        "bottom_10_label_indices": bottom_labels,
        "label_counts_top10": [int(per_label_counts[i]) for i in top_labels],
        "label_counts_bottom10": [int(per_label_counts[i]) for i in bottom_labels]
    }

def plot_hist(values, title, fname, xlabel="Value", bins=50):
    plt.figure(figsize=(8,4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname)
    plt.close()

def analyze_split(npz_path, split_name):
    X, y, acc = load_npz(npz_path)
    s_stats = safe_stats(X)
    l_stats = label_stats(y)

    # basic corruption checks
    nan_in_X = bool(np.isnan(X).any()) if np.issubdtype(X.dtype, np.floating) else False
    nan_in_y = bool(np.isnan(y).any())
    all_zero_seq = int(((X != 0).sum(axis=1) == 0).sum())
    all_zero_label = int((y.sum(axis=1) == 0).sum())

    # plots
    seq_lengths = (X != 0).sum(axis=1)
    plot_hist(seq_lengths, f"{split_name} - Sequence Lengths", f"{split_name}_seq_lengths.png", "Sequence length")
    plot_hist(y.sum(axis=1), f"{split_name} - Labels per Sample", f"{split_name}_labels_per_sample.png", "Labels per sample")
    per_label_counts = y.sum(axis=0)
    # plot top 100 labels to avoid huge plots
    topk = np.sort(per_label_counts)[-100:]
    plot_hist(topk, f"{split_name} - Top 100 Label Frequencies", f"{split_name}_top100_label_freqs.png", "Counts")

    report = {
        "split": split_name,
        "path": str(npz_path),
        "safe_stats": s_stats,
        "label_stats": l_stats,
        "corruption_checks": {
            "nan_in_X": nan_in_X,
            "nan_in_y": nan_in_y,
            "all_zero_sequences": all_zero_seq,
            "all_zero_labels": all_zero_label
        }
    }
    # include accession overlap info if accessions present
    if acc is not None:
        report["has_accessions"] = True
        report["n_unique_accessions"] = int(len(set(acc.tolist())))
    else:
        report["has_accessions"] = False

    return report, acc

def main():
    splits = {
        "train": Path("data/processed/train_data.npz"),
        "val": Path("data/processed/val_data.npz"),
        "test": Path("data/processed/test_data.npz")
    }
    overall_report = {"splits": {}, "warnings": []}
    accessions = {}

    for name, p in splits.items():
        if not p.exists():
            overall_report["warnings"].append(f"{name} file missing: {p}")
            continue
        rep, acc = analyze_split(p, name)
        overall_report["splits"][name] = rep
        accessions[name] = acc

    # check overlap between splits if accessions available
    if any(len(v) > 0 for v in accessions.values()):
        acc_sets = {k: set((v.tolist() if v is not None else [])) for k, v in accessions.items()}
        # pairwise overlaps
        overlaps = {}
        keys = list(acc_sets.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a, b = keys[i], keys[j]
                ov = len(acc_sets[a].intersection(acc_sets[b]))
                overlaps[f"{a}_vs_{b}"] = ov
                if ov > 0:
                    overall_report["warnings"].append(f"{ov} accessions overlap between {a} and {b}")
        overall_report["accession_overlaps"] = overlaps

    # quick pass/fail heuristics
    # thresholds (tunable)
    thresholds = {
        "min_avg_labels_per_sample": 1.0,   # expect >1 for multi-label task
        "max_zero_label_samples_fraction": 0.01,  # <1% zero-label samples
        "max_top1_label_share": 0.25  # if top label accounts >25% of all positive labels -> heavy imbalance
    }

    # evaluate train stats
    if "train" in overall_report["splits"]:
        train_ls = overall_report["splits"]["train"]["label_stats"]
        total_label_counts = np.sum(load_npz(splits["train"])[1])
        # top label share
        if total_label_counts > 0:
            top1 = train_ls["label_counts_top10"][0]
            share = top1 / total_label_counts
            if share > thresholds["max_top1_label_share"]:
                overall_report["warnings"].append(
                    f"Top label accounts for {share:.2%} of total positive labels (>{thresholds['max_top1_label_share']*100}%)"
                )
        # avg labels check
        if train_ls["avg_labels_per_sample"] < thresholds["min_avg_labels_per_sample"]:
            overall_report["warnings"].append("Average labels per sample is very low in train set.")
        # zero label samples fraction
        n_zero = overall_report["splits"]["train"]["label_stats"]["samples_with_zero_labels"]
        frac_zero = n_zero / overall_report["splits"]["train"]["label_stats"]["n_samples"]
        if frac_zero > thresholds["max_zero_label_samples_fraction"]:
            overall_report["warnings"].append(
                f"{frac_zero:.2%} samples in train set have zero labels (>{thresholds['max_zero_label_samples_fraction']*100}%)"
            )

    # save report
    out_json = REPORT_DIR / "data_quality_report.json"
    with open(out_json, "w") as f:
        json.dump(overall_report, f, indent=2)

    print(f"Data quality check complete. Report saved to: {out_json}")
    if overall_report["warnings"]:
        print("Warnings found:")
        for w in overall_report["warnings"]:
            print(" -", w)
    else:
        print("No major warnings found. Data looks OK for training.")

if __name__ == "__main__":
    main()
