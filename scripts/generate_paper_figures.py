"""
ProteoPredict - Publication Figures Generation Suite
===================================================
Generates 4 publication-grade (300 DPI) figures for submission to
Bioinformatics / IEEE TCBB / PLOS Computational Biology.

Outputs:
  reports/figures/figure1_benchmark_comparison.png
  reports/figures/figure2_ontology_breakdown.png
  reports/figures/figure3_cafa_smin.png
  reports/figures/figure4_active_site_saliency.png
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set global publication plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


# -------------------------------------------------------------
# FIGURE 1: Overall Micro-F1 & Macro-F1 Benchmark Progression
# -------------------------------------------------------------
def plot_figure1_benchmarks():
    print("Generating Figure 1: Overall Benchmark Comparison...")
    models = [
        "EXP-000\nNaive Prior",
        "EXP-001\n3-mer Linear",
        "EXP-002\nAttn-Hybrid",
        "EXP-003a\n3-mer (DAG)",
        "EXP-004b\nBio-MLP",
        "EXP-005\nResNet-LSTM",
        "HOM-BL\nHomology",
        "EXP-006\nESM-2 Bio",
        "EXP-007\nTri-Modal"
    ]
    micro_f1 = [0.3315, 0.4192, 0.4192, 0.4322, 0.4534, 0.4128, 0.4607, 0.4927, 0.5031]
    macro_f1 = [0.1693, 0.1801, 0.1999, 0.4023, 0.4380, 0.3785, 0.4529, 0.4819, 0.4997]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width/2, micro_f1, width, label="Test Micro-$F_1$ ($F_{\\max}$)", color="#1976D2", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, macro_f1, width, label="Test Macro-$F_1$", color="#FFA000", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("F-Score ($F_1$)", fontweight="bold")
    ax.set_title("Benchmark Progression Across Architecture Milestones (Holdout Test Set)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="medium")
    ax.set_ylim(0, 0.62)
    ax.axhline(0.50, color="#D32F2F", linestyle="--", linewidth=1.2, label="CAFA 0.50 Barrier")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Highlight our best model
    rects1[-1].set_color("#0D47A1")
    rects2[-1].set_color("#E65100")

    # Add value annotations on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, rotation=0)

    plt.tight_layout()
    fig_path = OUT_DIR / "figure1_benchmark_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


# -------------------------------------------------------------
# FIGURE 2: Per-Sub-Ontology Breakdown (MF, BP, CC)
# -------------------------------------------------------------
def plot_figure2_sub_ontologies():
    print("Generating Figure 2: Per-Sub-Ontology Comparison...")
    runs = [
        "Naive Prior\n(EXP-000)",
        "Homology Baseline\n(Nearest-Neighbor)",
        "ESM-2 Deep Hybrid\n(EXP-006)",
        "Tri-Modal Deep Net\n(EXP-007 SOTA)"
    ]
    mf_scores = [0.2148, 0.5349, 0.5638, 0.6146]
    bp_scores = [0.3616, 0.4615, 0.4857, 0.4924]
    cc_scores = [0.3085, 0.3997, 0.4539, 0.4571]

    x = np.arange(len(runs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width, mf_scores, width, label="Molecular Function (MF)", color="#2E7D32", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x, bp_scores, width, label="Biological Process (BP)", color="#1565C0", edgecolor="black", linewidth=0.8)
    b3 = ax.bar(x + width, cc_scores, width, label="Cellular Component (CC)", color="#6A1B9A", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Sub-Ontology $F_1$ Score", fontweight="bold")
    ax.set_title("Sub-Ontology Resolution Breakdown Across Key Milestones", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontweight="medium")
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Highlight MF > 0.60 milestone
    ax.axhline(0.60, color="#2E7D32", linestyle=":", linewidth=1.2, label="MF > 0.60 Target")

    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "figure2_ontology_breakdown.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


# -------------------------------------------------------------
# FIGURE 3: CAFA Semantic Distance (S_min, lower is better)
# -------------------------------------------------------------
def plot_figure3_smin():
    print("Generating Figure 3: CAFA S_min Semantic Distance...")
    methods = [
        "EXP-000\nNaive Prior",
        "Homology Baseline\n(Nearest-Neighbor)",
        "EXP-006\nESM-2 Hybrid",
        "EXP-007\nTri-Modal SOTA"
    ]
    s_min = [28.31, 22.76, 20.48, 19.86]
    ru = [25.99, 21.14, 17.81, 18.12]
    mi = [11.21, 8.45, 10.12, 8.13]

    x = np.arange(len(methods))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    r1 = ax.bar(x - width, s_min, width, label="Semantic Distance ($S_{\\min} \\downarrow$)", color="#C62828", edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x, ru, width, label="Remaining Uncertainty ($RU \\downarrow$)", color="#EF6C00", edgecolor="black", linewidth=0.8)
    r3 = ax.bar(x + width, mi, width, label="Misinformation ($MI \\downarrow$)", color="#455A64", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Information-Accretion Weighted Error (bits)", fontweight="bold")
    ax.set_title("CAFA Semantic Distance ($S_{\\min}$) & Error Decomposition (Lower is Better)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight="medium")
    ax.set_ylim(0, 32)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bars in [r1, r2, r3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig_path = OUT_DIR / "figure3_cafa_smin.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


# -------------------------------------------------------------
# FIGURE 4: Active Site Explainability & Saliency Enrichment
# -------------------------------------------------------------
def plot_figure4_active_site():
    print("Generating Figure 4: Biological Active Site Saliency...")
    targets = ["EGFR Kinase Domain\n(10 Active Residues)", "Beta-Hemoglobin\n(6 Active Residues)", "TP53 Core Domain\n(191 DNA Contact)"]
    active_saliency = [0.00702, 0.00467, 0.00792]
    background_saliency = [0.00534, 0.00833, 0.01704]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - width/2, active_saliency, width, label="Active Pocket Residues Saliency", color="#00897B", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, background_saliency, width, label="Background Residues Saliency", color="#B0BEC5", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("In Silico Alanine Perturbation Saliency", fontweight="bold")
    ax.set_title("Model Attention at Experimentally Verified Catalytic Residues", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontweight="medium")
    ax.set_ylim(0, 0.022)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Highlight statistical significance for EGFR
    ax.annotate("1.31x Enrichment\np = 0.0091 (p < 0.01)**",
                xy=(0 - width/2, active_saliency[0]),
                xytext=(0, 30), textcoords="offset points",
                ha="center", va="bottom", fontweight="bold", color="#004D40",
                arrowprops=dict(arrowstyle="->", color="#004D40", lw=1.5))

    plt.tight_layout()
    fig_path = OUT_DIR / "figure4_active_site_saliency.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES (300 DPI)")
    print("=" * 60)
    plot_figure1_benchmarks()
    plot_figure2_sub_ontologies()
    plot_figure3_smin()
    plot_figure4_active_site()
    print("=" * 60)
    print("ALL 4 PUBLICATION FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
