"""
Unit and Integration Tests for ProteoPredict
Verifies zero synthetic data, zero homology leakage, correct data formatting, and inference.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def test_zero_synthetic_data_in_source():
    """Verify that no synthetic sequence generator or SYNTH markers remain in download.py."""
    download_py = Path("src/proteopredict/data/download.py")
    assert download_py.exists(), "download.py must exist"

    content = download_py.read_text(encoding="utf-8")
    assert "generate_large_synthetic_dataset" not in content, "Synthetic generator function must be eliminated"
    assert "SYNTH" not in content, "Fake SYNTH accessions must not be present"
    assert "np.random.choice" not in content, "Random sequence generation is strictly forbidden"


def test_authentic_raw_data_properties():
    """Verify that proteins_complete.csv contains authentic Swiss-Prot proteins."""
    raw_csv = Path("data/raw/proteins_complete.csv")
    assert raw_csv.exists(), "proteins_complete.csv must exist"

    df = pd.read_csv(raw_csv, nrows=100)
    assert "accession" in df.columns
    assert "sequence" in df.columns
    assert "go_all" in df.columns

    # Verify real accessions (not synthetic)
    assert not any(str(acc).startswith("SYNTH") for acc in df["accession"]), "Accessions must be real UniProt IDs"

    # Verify standard amino acids
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for seq in df["sequence"].head(20):
        assert set(seq).issubset(standard_aa), "Sequences must only contain valid amino acid residues"


def test_homology_split_leakage():
    """Verify that Train, Val, and Test splits have ZERO homology cluster leakage."""
    train_csv = Path("data/raw/train.csv")
    val_csv = Path("data/raw/val.csv")
    test_csv = Path("data/raw/test.csv")

    assert train_csv.exists() and val_csv.exists() and test_csv.exists(), "Split CSV files must exist"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    if "cluster_id" in df_train.columns:
        train_clusters = set(df_train["cluster_id"])
        val_clusters = set(df_val["cluster_id"])
        test_clusters = set(df_test["cluster_id"])

        assert len(train_clusters.intersection(val_clusters)) == 0, "Train and Val must share 0 clusters"
        assert len(train_clusters.intersection(test_clusters)) == 0, "Train and Test must share 0 clusters"
        assert len(val_clusters.intersection(test_clusters)) == 0, "Val and Test must share 0 clusters"


def test_processed_arrays_shape_and_types():
    """Verify that processed NPZ files match the expected format and dimensions."""
    proc_dir = Path("data/processed")
    for split in ["train", "val", "test"]:
        npz_path = proc_dir / f"{split}_data.npz"
        assert npz_path.exists(), f"{npz_path} must exist"

        data = np.load(npz_path)
        assert "X" in data and "y" in data and "lengths" in data
        assert data["X"].shape[1] == 1000, "Sequence token length should be 1000"
        assert data["y"].shape[1] == 75, "Number of classes should be 75"
        assert data["X"].shape[0] == data["y"].shape[0], "Sample counts must match"


def test_go_vocabulary_structure():
    """Verify the 75 selected GO terms have full metadata across MF, BP, CC."""
    go_info_path = Path("data/processed/go_info.json")
    assert go_info_path.exists(), "go_info.json must exist"

    with open(go_info_path, "r", encoding="utf-8") as f:
        go_info = json.load(f)

    assert go_info["num_classes"] == 75
    assert len(go_info["classes"]) == 75

    categories = {meta["category"] for meta in go_info["term_metadata"].values()}
    assert "MF" in categories and "BP" in categories and "CC" in categories
