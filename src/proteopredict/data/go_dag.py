"""
ProteoPredict - Gene Ontology DAG Propagation Engine
Implements the True Path Rule across the Gene Ontology Directed Acyclic Graph (DAG)
using the official Gene Ontology Consortium go-basic.obo specification.
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict, deque
import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProteoPredict.GODAG")


class GeneOntologyDAG:
    """Parses and computes transitive ancestor closures for Gene Ontology terms."""

    OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"

    def __init__(self, obo_path: str = "data/raw/go-basic.obo"):
        self.obo_path = Path(obo_path)
        self.terms = {}  # term_id -> {name, namespace, parents, is_obsolete}
        self.ancestors_cache = {}  # term_id -> set of all ancestor term_ids (including self)
        self.namespace_map = {
            "molecular_function": "MF",
            "biological_process": "BP",
            "cellular_component": "CC"
        }

    def download_obo(self, timeout: int = 300) -> Path:
        """Download official go-basic.obo from Gene Ontology Consortium."""
        if self.obo_path.exists() and self.obo_path.stat().st_size > 10 * 1024 * 1024:
            logger.info(f"OBO file already exists at {self.obo_path} ({self.obo_path.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
            return self.obo_path

        self.obo_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading go-basic.obo from {self.OBO_URL}...")
        t0 = time.time()

        with requests.get(self.OBO_URL, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(self.obo_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading OBO: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)...", end="", flush=True)

        print()
        logger.info(f"Downloaded go-basic.obo in {time.time() - t0:.1f}s to {self.obo_path}")
        return self.obo_path

    def load_and_parse_obo(self):
        """Parse go-basic.obo into terms dictionary with parent links (is_a, part_of)."""
        if not self.obo_path.exists():
            self.download_obo()

        logger.info(f"Parsing Gene Ontology OBO structure from {self.obo_path}...")
        t0 = time.time()

        current_term = None
        in_term_stanza = False

        with open(self.obo_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == "[Term]":
                    if current_term and not current_term.get("is_obsolete", False):
                        self.terms[current_term["id"]] = current_term
                    current_term = {"parents": set(), "is_obsolete": False}
                    in_term_stanza = True
                    continue
                elif line.startswith("[") and line.endswith("]"):
                    if current_term and not current_term.get("is_obsolete", False):
                        self.terms[current_term["id"]] = current_term
                    current_term = None
                    in_term_stanza = False
                    continue

                if in_term_stanza and current_term is not None:
                    if line.startswith("id: GO:"):
                        current_term["id"] = line.split("id: ")[1].split()[0]
                    elif line.startswith("name: "):
                        current_term["name"] = line.split("name: ")[1].strip()
                    elif line.startswith("namespace: "):
                        raw_ns = line.split("namespace: ")[1].strip()
                        current_term["namespace"] = self.namespace_map.get(raw_ns, raw_ns)
                    elif line.startswith("is_obsolete: true"):
                        current_term["is_obsolete"] = True
                    elif line.startswith("is_a: GO:"):
                        parent_id = line.split("is_a: ")[1].split()[0]
                        current_term["parents"].add(parent_id)
                    elif line.startswith("relationship: part_of GO:"):
                        parent_id = line.split("relationship: part_of ")[1].split()[0]
                        current_term["parents"].add(parent_id)

        # Catch trailing term
        if current_term and not current_term.get("is_obsolete", False):
            self.terms[current_term["id"]] = current_term

        logger.info(f"Parsed {len(self.terms):,} valid active Gene Ontology terms in {time.time() - t0:.2f}s.")

    def get_ancestors(self, term_id: str) -> set[str]:
        """
        Compute full ancestor closure (including self) for a given GO term.
        Traverses upward along 'is_a' and 'part_of' edges to root terms.
        """
        if term_id in self.ancestors_cache:
            return self.ancestors_cache[term_id]

        if term_id not in self.terms:
            # Term not found in current OBO, return self only
            return {term_id}

        ancestors = set()
        queue = deque([term_id])

        while queue:
            curr = queue.popleft()
            if curr not in ancestors:
                ancestors.add(curr)
                if curr in self.terms:
                    for parent in self.terms[curr]["parents"]:
                        if parent not in ancestors:
                            queue.append(parent)

        self.ancestors_cache[term_id] = ancestors
        return ancestors

    def propagate_term_list(self, term_list: list[str]) -> set[str]:
        """Propagate a list of specific leaf terms up to all biological ancestors."""
        closure = set()
        for term in term_list:
            if term:
                closure.update(self.get_ancestors(term.strip()))
        return closure

    def propagate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply True Path Rule to an entire protein dataframe across MF, BP, and CC branches.
        Generates propagated GO term columns: go_mf_dag, go_bp_dag, go_cc_dag, go_all_dag.
        """
        if not self.terms:
            self.load_and_parse_obo()

        logger.info(f"Propagating annotations across Gene Ontology DAG for {len(df):,} proteins...")
        t0 = time.time()

        mf_roots = {"GO:0003674"}  # molecular_function root
        bp_roots = {"GO:0008150"}  # biological_process root
        cc_roots = {"GO:0005575"}  # cellular_component root
        all_roots = mf_roots | bp_roots | cc_roots

        go_mf_dag = []
        go_bp_dag = []
        go_cc_dag = []
        go_all_dag = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="DAG Propagation"):
            # Raw terms
            raw_mf = [t.strip() for t in str(row.get("go_mf", "")).split(";") if t.strip() and t.strip() != "nan"]
            raw_bp = [t.strip() for t in str(row.get("go_bp", "")).split(";") if t.strip() and t.strip() != "nan"]
            raw_cc = [t.strip() for t in str(row.get("go_cc", "")).split(";") if t.strip() and t.strip() != "nan"]

            # Transitive closures
            prop_mf = self.propagate_term_list(raw_mf) - all_roots
            prop_bp = self.propagate_term_list(raw_bp) - all_roots
            prop_cc = self.propagate_term_list(raw_cc) - all_roots
            prop_all = (prop_mf | prop_bp | prop_cc)

            go_mf_dag.append(";".join(sorted(list(prop_mf))))
            go_bp_dag.append(";".join(sorted(list(prop_bp))))
            go_cc_dag.append(";".join(sorted(list(prop_cc))))
            go_all_dag.append(";".join(sorted(list(prop_all))))

        df["go_mf_dag"] = go_mf_dag
        df["go_bp_dag"] = go_bp_dag
        df["go_cc_dag"] = go_cc_dag
        df["go_all_dag"] = go_all_dag

        logger.info(f"DAG propagation complete in {time.time() - t0:.2f}s.")
        return df


def main():
    parser = argparse.ArgumentParser(description="Propagate Gene Ontology annotations up the DAG.")
    parser.add_argument("--input_csv", type=str, default="data/raw/proteins_complete.csv", help="Input protein dataset CSV")
    parser.add_argument("--output_csv", type=str, default="data/raw/proteins_complete_dag.csv", help="Output propagated CSV")
    parser.add_argument("--obo_path", type=str, default="data/raw/go-basic.obo", help="Path to go-basic.obo")
    args = parser.parse_args()

    dag = GeneOntologyDAG(obo_path=args.obo_path)
    dag.download_obo()
    dag.load_and_parse_obo()

    df = pd.read_csv(args.input_csv)
    df_dag = dag.propagate_dataframe(df)
    df_dag.to_csv(args.output_csv, index=False)
    logger.info(f"Saved DAG-propagated dataset to {args.output_csv}")


if __name__ == "__main__":
    main()
