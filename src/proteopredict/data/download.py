"""
ProteoPredict - Authentic Data Download Module
Acquires high-quality, manually curated proteins from UniProtKB / Swiss-Prot
with verified experimental and computational Gene Ontology (GO) annotations.

Zero synthetic data generation, zero random sequences.
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.Download")


class ProteoDataDownloader:
    """Authentic data acquisition pipeline for UniProtKB/Swiss-Prot curated proteins."""

    UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
    UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

    STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_query_count(self, query: str) -> int:
        """Query UniProt REST API to get total number of matching reviewed entries."""
        try:
            params = {"query": query, "size": 0}
            resp = requests.get(self.UNIPROT_SEARCH_URL, params=params, timeout=30)
            if resp.status_code == 200:
                total = int(resp.headers.get("x-total-results", 0))
                return total
        except Exception as e:
            logger.warning(f"Could not retrieve entry count: {e}")
        return 0

    def download_uniprot_stream(
        self,
        query: str = "reviewed:true AND organism_id:9606",
        max_proteins: int = None,
        chunk_size: int = 1024 * 64,
        timeout: int = 300,
    ) -> Path:
        """
        Download reviewed UniProt proteins via the UniProtKB REST stream API.
        Zero synthetic fallback: fails loudly if network or UniProt is unavailable.
        """
        raw_tsv_path = self.output_dir / "uniprot_raw.tsv"
        total_available = self.get_query_count(query)
        logger.info(f"Target UniProt Query: '{query}'")
        logger.info(f"Total matching entries reported by UniProt: {total_available}")

        params = {
            "query": query,
            "format": "tsv",
            "fields": (
                "accession,id,protein_name,gene_names,organism_name,length,"
                "sequence,go_id,go_p,go_c,go_f"
            ),
        }

        if max_proteins and max_proteins > 0:
            params["size"] = str(max_proteins)

        logger.info(f"Streaming data from {self.UNIPROT_STREAM_URL}...")
        t0 = time.time()

        try:
            with requests.get(
                self.UNIPROT_STREAM_URL, params=params, stream=True, timeout=timeout
            ) as response:
                response.raise_for_status()

                bytes_downloaded = 0
                with open(raw_tsv_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            mb = bytes_downloaded / (1024 * 1024)
                            print(f"\rDownloaded {mb:.2f} MB...", end="", flush=True)

            print()
            elapsed = time.time() - t0
            file_mb = raw_tsv_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully downloaded {file_mb:.2f} MB to {raw_tsv_path} in {elapsed:.1f}s"
            )
            return raw_tsv_path

        except Exception as e:
            logger.error(f"UniProt download failed: {e}")
            raise RuntimeError(
                f"Failed to download authentic UniProt protein data: {e}. "
                "Synthetic data generation is strictly prohibited in ProteoPredict."
            )

    @staticmethod
    def _parse_go_field(field_value: str):
        """
        Parses UniProt GO fields formatted as 'term name [GO:XXXXXXX]; term2 [GO:YYYYYYY]'
        Returns a tuple of (list_of_go_ids, dict_of_go_id_to_name).
        """
        if not field_value or pd.isna(field_value):
            return [], {}

        go_ids = []
        name_map = {}
        # Pattern matches: term name [GO:0000000]
        pattern = re.compile(r"^(.*?)\s*\[(GO:\d+)\]$")

        entries = str(field_value).split(";")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            match = pattern.match(entry)
            if match:
                name, go_id = match.group(1).strip(), match.group(2).strip()
                go_ids.append(go_id)
                name_map[go_id] = name
            elif "GO:" in entry:
                # Fallback if bare GO ID is present
                go_matches = re.findall(r"GO:\d+", entry)
                for gid in go_matches:
                    go_ids.append(gid)
                    name_map[gid] = entry

        return go_ids, name_map

    def process_and_filter_data(
        self,
        raw_tsv_path: Path,
        min_length: int = 50,
        max_length: int = 1500,
        max_proteins: int = None,
    ) -> pd.DataFrame:
        """
        Clean, parse, and filter authentic protein sequences and GO annotations.
        """
        logger.info(f"Reading raw UniProt TSV from {raw_tsv_path}...")
        df = pd.read_csv(raw_tsv_path, sep="\t", nrows=max_proteins)
        initial_count = len(df)
        logger.info(f"Loaded {initial_count} raw UniProt entries.")

        # Standardize column names
        column_mapping = {
            "Entry": "accession",
            "Entry Name": "entry_name",
            "Protein names": "protein_name",
            "Gene Names": "gene_names",
            "Organism": "organism",
            "Length": "length",
            "Sequence": "sequence",
            "Gene Ontology IDs": "go_all",
            "Gene Ontology (biological process)": "go_bp_raw",
            "Gene Ontology (cellular component)": "go_cc_raw",
            "Gene Ontology (molecular function)": "go_mf_raw",
        }
        df = df.rename(columns=column_mapping)

        # 1. Filter missing sequence or length
        df = df[df["sequence"].notna()].copy()
        df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

        # 2. Sequence length filtering
        df = df[(df["sequence"].str.len() >= min_length) & (df["sequence"].str.len() <= max_length)].copy()
        logger.info(
            f"Retained {len(df)} proteins within sequence length bounds [{min_length}, {max_length}]."
        )

        # 3. Filter non-standard amino acid sequences (preserve standard 20 amino acids)
        def is_valid_sequence(seq):
            return set(seq).issubset(self.STANDARD_AMINO_ACIDS)

        valid_mask = df["sequence"].apply(is_valid_sequence)
        num_invalid = (~valid_mask).sum()
        df = df[valid_mask].copy()
        logger.info(
            f"Filtered {num_invalid} sequences containing non-standard amino acids (U, O, B, Z, X). {len(df)} proteins remaining."
        )

        # 4. Parse GO terms and term descriptions
        global_go_names = {}
        go_bp_col = []
        go_cc_col = []
        go_mf_col = []
        go_all_col = []

        logger.info("Parsing structured Gene Ontology annotations (BP, CC, MF)...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing GO"):
            bp_ids, bp_names = self._parse_go_field(row.get("go_bp_raw", ""))
            cc_ids, cc_names = self._parse_go_field(row.get("go_cc_raw", ""))
            mf_ids, mf_names = self._parse_go_field(row.get("go_mf_raw", ""))

            global_go_names.update(bp_names)
            global_go_names.update(cc_names)
            global_go_names.update(mf_names)

            # Combined list of all unique GO IDs
            all_ids = list(dict.fromkeys(bp_ids + cc_ids + mf_ids))

            go_bp_col.append(";".join(bp_ids))
            go_cc_col.append(";".join(cc_ids))
            go_mf_col.append(";".join(mf_ids))
            go_all_col.append(";".join(all_ids))

        df["go_bp"] = go_bp_col
        df["go_cc"] = go_cc_col
        df["go_mf"] = go_mf_col
        df["go_all"] = go_all_col

        # 5. Filter out proteins that have zero valid GO terms across all categories
        has_go = df["go_all"].str.len() > 0
        df = df[has_go].copy()
        logger.info(f"Retained {len(df)} proteins with at least one validated GO term.")

        # Clean up temporary raw columns
        for col in ["go_bp_raw", "go_cc_raw", "go_mf_raw"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Save processed files
        complete_csv = self.output_dir / "proteins_complete.csv"
        df.to_csv(complete_csv, index=False)
        logger.info(f"Saved complete cleaned authentic dataset: {complete_csv}")

        # Save comprehensive GO names dictionary
        go_names_path = self.output_dir / "go_names.json"
        with open(go_names_path, "w", encoding="utf-8") as f:
            json.dump(global_go_names, f, indent=2)
        logger.info(f"Saved {len(global_go_names)} GO term descriptions to {go_names_path}")

        # Summary statistics
        all_unique_terms = set(global_go_names.keys())
        vocab_path = self.output_dir / "go_vocabulary.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_terms": len(all_unique_terms),
                    "all_terms": sorted(list(all_unique_terms)),
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 70)
        print("AUTHENTIC DATASET SUMMARY")
        print("=" * 70)
        print(f"Total Reviewed Proteins:   {len(df):,}")
        print(f"Average Sequence Length:   {df['length'].mean():.1f} aa")
        print(f"Sequence Length Range:     {df['length'].min()} - {df['length'].max()} aa")
        print(f"Unique Curated GO Terms:   {len(all_unique_terms):,}")
        print("=" * 70 + "\n")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Download authentic reviewed protein sequences from UniProtKB."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="reviewed:true AND organism_id:9606",
        help="UniProt query (default: human reviewed Swiss-Prot proteins)",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Maximum proteins to ingest (default: all reviewed in query)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=50,
        help="Minimum sequence length (default: 50)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1500,
        help="Maximum sequence length (default: 1500)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save raw data files",
    )

    args = parser.parse_args()

    downloader = ProteoDataDownloader(output_dir=args.output_dir)
    raw_path = downloader.download_uniprot_stream(
        query=args.query, max_proteins=args.max_proteins
    )
    downloader.process_and_filter_data(
        raw_path,
        min_length=args.min_length,
        max_length=args.max_length,
        max_proteins=args.max_proteins,
    )


if __name__ == "__main__":
    main()