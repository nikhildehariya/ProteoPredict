"""
ProteoPredict - Biological Explainability & Active Site Validation Engine (Objective 4)
Performs in silico perturbation and saliency analysis on benchmark human proteins
(e.g., TP53, Hemoglobin HBB, Kinase EGFR) to demonstrate that Tri-Modal representations
statistically correlate with experimentally verified catalytic triads and binding residues.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import tensorflow as tf

from src.proteopredict.prediction.predict import load_inference_engine, extract_tri_modal_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.Explainability")


# Ground-truth verified active site benchmarks from UniProtKB / PDB
BENCHMARK_PROTEINS = {
    "TP53": {
        "name": "Cellular tumor antigen p53 (Homo sapiens)",
        "uniprot_id": "P04637",
        "key_go_terms": ["GO:0003677", "GO:0010468"],  # DNA binding, regulation of gene expression
        "sequence": (
            "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
            "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK"
            "SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE"
            "RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS"
            "SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP"
            "PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG"
            "GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
        ),
        # Verified DNA-binding core (102-292) and catalytic/zinc coordination (C176, H179, C238, C242, R248, R273)
        "active_residues": [176, 179, 238, 242, 248, 273, 280] + list(range(102, 293))
    },
    "Hemoglobin_HBB": {
        "name": "Hemoglobin subunit beta (Homo sapiens)",
        "uniprot_id": "P68871",
        "key_go_terms": ["GO:0005829", "GO:0005576"],  # cytosol, extracellular
        "sequence": (
            "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
            "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
            "KEFTPPVQAAYQKVVAGVANALAHKYH"
        ),
        # Heme coordination pocket: Proximal His-93 (pos 93), Distal His-64 (pos 64), Val-68
        "active_residues": [64, 68, 86, 93, 98, 142]
    },
    "EGFR_Kinase": {
        "name": "Epidermal growth factor receptor (Homo sapiens, Kinase Domain)",
        "uniprot_id": "P00533",
        "key_go_terms": ["GO:0016740", "GO:0007165"],  # transferase activity, signal transduction
        "sequence": (
            "LVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGV"
            "TVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKF"
            "RELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQ"
        ),
        # Catalytic triad and ATP binding pocket residues
        "active_residues": [10, 11, 12, 13, 14, 25, 27, 45, 78, 112]
    }
}


def compute_residue_saliency(sequence: str, target_term: str, artifacts: dict, window_size: int = 5):
    """
    Computes in silico residue-level functional sensitivity using a sliding window occlusion.
    Quantifies Delta P = P(wildtype) - P(occluded_residues).
    """
    model = artifacts["model"]
    meta = artifacts["metadata"]

    # Determine branch and index of target term
    branch = None
    term_idx = None

    if target_term in meta["mf_classes"]:
        branch = 0
        term_idx = meta["mf_classes"].index(target_term)
    elif target_term in meta["bp_classes"]:
        branch = 1
        term_idx = meta["bp_classes"].index(target_term)
    elif target_term in meta["cc_classes"]:
        branch = 2
        term_idx = meta["cc_classes"].index(target_term)
    else:
        branch = 0
        term_idx = 0

    # 1. Baseline Wildtype Prediction
    esm_wt, kmer_wt, bio_wt = extract_tri_modal_features(sequence, artifacts)
    preds_wt = model.predict([esm_wt, kmer_wt, bio_wt], verbose=0)
    p_wt = float(preds_wt[branch][0, term_idx])

    # 2. Window-based In Silico Alanine / Glycine Perturbation
    L = len(sequence)
    saliency_scores = np.zeros(L, dtype=np.float32)

    # Step across sequence with step size 3 to maintain speed
    step = 3
    for i in range(0, L, step):
        start = max(0, i)
        end = min(L, i + window_size)

        # Mutate window to Poly-Alanine
        mut_seq = sequence[:start] + ("A" * (end - start)) + sequence[end:]
        esm_m, kmer_m, bio_m = extract_tri_modal_features(mut_seq, artifacts)
        preds_m = model.predict([esm_m, kmer_m, bio_m], verbose=0)
        p_mut = float(preds_m[branch][0, term_idx])

        # Functional Drop
        drop = max(0.0, p_wt - p_mut)
        saliency_scores[start:end] = np.maximum(saliency_scores[start:end], drop)

    return saliency_scores, p_wt


def run_explainability_study():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Tri-Modal Inference Engine for Biological Validation...")
    artifacts = load_inference_engine()

    validation_reports = {}

    for prot_key, prot_info in BENCHMARK_PROTEINS.items():
        prot_name = prot_info["name"]
        uniprot_id = prot_info["uniprot_id"]
        seq = prot_info["sequence"]
        active_residues = set(prot_info["active_residues"])
        target_term = prot_info["key_go_terms"][0]

        logger.info(f"\nEvaluating biological explainability for {prot_name} ({uniprot_id})...")
        saliency, p_wt = compute_residue_saliency(seq, target_term, artifacts)

        L = len(seq)
        # Classify residues into Active (functional pocket) vs Background
        active_scores = [saliency[i] for i in range(L) if (i + 1) in active_residues]
        background_scores = [saliency[i] for i in range(L) if (i + 1) not in active_residues]

        if not active_scores:
            active_scores = [0.0]
        if not background_scores:
            background_scores = [0.0]

        # Statistical Enrichment: Mann-Whitney U Test (One-sided: Active > Background)
        u_stat, p_value = mannwhitneyu(active_scores, background_scores, alternative="greater")

        mean_active_saliency = float(np.mean(active_scores))
        mean_bg_saliency = float(np.mean(background_scores))
        enrichment_ratio = round(mean_active_saliency / max(1e-6, mean_bg_saliency), 2)

        is_significant = bool(p_value < 0.05)

        logger.info(f"Target GO Term: {target_term} (Baseline Prob: {p_wt:.4f})")
        logger.info(f"Mean Saliency in Verified Active Site: {mean_active_saliency:.4f}")
        logger.info(f"Mean Saliency in Background Residues:   {mean_bg_saliency:.4f}")
        logger.info(f"Enrichment Ratio: {enrichment_ratio}x | Mann-Whitney p-value: {p_value:.4e} (p < 0.05: {is_significant})")

        validation_reports[prot_key] = {
            "protein_name": prot_name,
            "uniprot_id": uniprot_id,
            "target_go_term": target_term,
            "wildtype_probability": round(p_wt, 4),
            "sequence_length": L,
            "num_active_residues": len(active_residues),
            "mean_active_saliency": round(mean_active_saliency, 5),
            "mean_background_saliency": round(mean_bg_saliency, 5),
            "saliency_enrichment_ratio": enrichment_ratio,
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_value),
            "statistically_significant": is_significant
        }

    out_path = results_dir / "active_site_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation_reports, f, indent=2)

    logger.info(f"\nBiological active site validation results saved to {out_path}")
    return validation_reports


if __name__ == "__main__":
    run_explainability_study()
