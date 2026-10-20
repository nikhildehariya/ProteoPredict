"""
ProteoPredict - ESM-2 Pre-Trained Protein Language Model Feature Extractor
Extracts 320-dimensional evolutionary sequence embeddings using Meta AI's ESM-2 (8M parameters)
to provide deep evolutionary and structural priors for Gene Ontology function prediction.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import esm
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.ESM2")


def extract_esm2_embeddings(
    fasta_or_df,
    output_path: Path,
    batch_size: int = 16,
    max_len: int = 1000
) -> np.ndarray:
    """
    Extracts 320D mean-pooled sequence embeddings from ESM-2 (esm2_t6_8M_UR50D).
    """
    if output_path.exists():
        logger.info(f"Loading pre-computed ESM-2 embeddings from {output_path}...")
        return np.load(output_path)

    logger.info("Initializing ESM-2 (esm2_t6_8M_UR50D, 8M parameters)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    if isinstance(fasta_or_df, (pd.DataFrame, pd.Series)):
        if isinstance(fasta_or_df, pd.DataFrame):
            data_records = list(zip(fasta_or_df["accession"], fasta_or_df["sequence"]))
        else:
            data_records = [(f"seq_{i}", s) for i, s in enumerate(fasta_or_df)]
    else:
        data_records = fasta_or_df

    logger.info(f"Extracting ESM-2 embeddings for {len(data_records)} sequences...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data_records), batch_size), desc="ESM-2 Embedding Extraction"):
            batch_slice = data_records[i : i + batch_size]
            # Truncate sequences to max_len to fit context
            batch_clean = [(str(pid), str(seq)[:max_len].upper()) for pid, seq in batch_slice]

            batch_labels, batch_strs, batch_tokens = batch_converter(batch_clean)
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]

            # Mean-pool over sequence length (excluding [CLS] and [EOS] tokens)
            for j, (_, seq) in enumerate(batch_clean):
                seq_len = len(seq)
                seq_rep = token_representations[j, 1 : seq_len + 1].mean(0).cpu().numpy()
                all_embeddings.append(seq_rep)

    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_arr)
    logger.info(f"Successfully saved ESM-2 embeddings (shape: {embeddings_arr.shape}) to {output_path}")

    return embeddings_arr


def run_batch_extraction():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    train_df = pd.read_csv(raw_dir / "train.csv")
    val_df = pd.read_csv(raw_dir / "val.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")

    logger.info("Starting ESM-2 feature extraction for Train, Val, and Test splits...")
    train_esm = extract_esm2_embeddings(train_df, processed_dir / "esm2_train.npy", batch_size=32)
    val_esm = extract_esm2_embeddings(val_df, processed_dir / "esm2_val.npy", batch_size=32)
    test_esm = extract_esm2_embeddings(test_df, processed_dir / "esm2_test.npy", batch_size=32)

    logger.info(f"ESM-2 Extraction Complete! Shapes: Train={train_esm.shape}, Val={val_esm.shape}, Test={test_esm.shape}")


if __name__ == "__main__":
    run_batch_extraction()
