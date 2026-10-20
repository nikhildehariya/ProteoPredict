"""
ProteoPredict - Biophysical & Physicochemical Feature Extraction Module
Extracts domain-level biochemical descriptors (composition, charge, hydrophobicity,
secondary structure propensity, dipeptide frequencies) to enrich sequence representations.
"""

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

# Kyte-Doolittle Hydrophobicity Scale
KD_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

# Charge at pH 7.4
CHARGE_AT_PH7 = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0
}


def extract_protein_biophysical_features(sequence: str) -> np.ndarray:
    """
    Extract a comprehensive 430-dimensional biophysical feature vector:
    - 20D Amino Acid Composition (AAC)
    - 400D Dipeptide Composition (DPC)
    - 10D Physicochemical Properties (MW, pI, charge, hydrophobicity, secondary structure)
    """
    clean_seq = "".join([aa for aa in str(sequence).upper() if aa in AA_TO_IDX])
    L = len(clean_seq)
    if L < 5:
        return np.zeros(430, dtype=np.float32)

    features = []

    # 1. 20D Amino Acid Composition
    counts = {aa: 0 for aa in STANDARD_AA}
    for aa in clean_seq:
        counts[aa] += 1
    aac = [counts[aa] / L for aa in STANDARD_AA]
    features.extend(aac)

    # 2. 400D Dipeptide Composition
    dipeptide_counts = np.zeros((20, 20), dtype=np.float32)
    for i in range(L - 1):
        idx1 = AA_TO_IDX[clean_seq[i]]
        idx2 = AA_TO_IDX[clean_seq[i+1]]
        dipeptide_counts[idx1, idx2] += 1
    dpc = (dipeptide_counts / (L - 1)).flatten().tolist()
    features.extend(dpc)

    # 3. 10D Physicochemical & Secondary Structure Properties
    try:
        analysed = ProteinAnalysis(clean_seq)
        mw = analysed.molecular_weight() / 10000.0  # normalized
        pi = analysed.isoelectric_point() / 14.0   # normalized
        aromaticity = analysed.aromaticity()
        instability = analysed.instability_index() / 100.0
        gravy = analysed.gravy() / 5.0  # hydropathicity
        helix, turn, sheet = analysed.secondary_structure_fraction()
    except Exception:
        mw = L * 110.0 / 10000.0
        pi = 0.5
        aromaticity = (counts["F"] + counts["W"] + counts["Y"]) / L
        instability = 0.4
        gravy = sum(KD_HYDROPHOBICITY.get(aa, 0) for aa in clean_seq) / (L * 5.0)
        helix, turn, sheet = 0.3, 0.2, 0.2

    net_charge = sum(CHARGE_AT_PH7.get(aa, 0) for aa in clean_seq) / L
    avg_hydro = sum(KD_HYDROPHOBICITY.get(aa, 0) for aa in clean_seq) / L

    features.extend([mw, pi, aromaticity, instability, gravy, helix, turn, sheet, net_charge, avg_hydro])

    return np.array(features, dtype=np.float32)


def batch_extract_features(sequences: list[str]) -> np.ndarray:
    """Extract biophysical features for a batch of protein sequences."""
    feat_matrix = [extract_protein_biophysical_features(seq) for seq in tqdm(sequences, desc="Extracting Biophysical Descriptors")]
    return np.array(feat_matrix, dtype=np.float32)
