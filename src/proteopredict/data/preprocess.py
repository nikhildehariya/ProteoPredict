"""
ProteoPredict - Data Preprocessing Module
Encodes sequences and prepares data for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import json

class ProteoDataPreprocessor:
    """Preprocess protein sequences and GO annotations"""
    
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Amino acid vocabulary (20 standard amino acids + padding)
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i+1 for i, aa in enumerate(self.amino_acids)}
        self.aa_to_int['X'] = 21  # Unknown amino acid
        self.aa_to_int['<PAD>'] = 0  # Padding token
        
        # Reverse mapping
        self.int_to_aa = {v: k for k, v in self.aa_to_int.items()}
        
        print(f"✓ Amino acid vocabulary: {len(self.aa_to_int)} tokens")
    
    def encode_sequences(self, sequences, max_length=1000):
        """
        Encode amino acid sequences to integer sequences
        
        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length (pad/truncate)
        
        Returns:
            numpy array of encoded sequences
        """
        print(f"\n{'='*70}")
        print("ENCODING SEQUENCES")
        print(f"{'='*70}")
        print(f"Number of sequences: {len(sequences)}")
        print(f"Max length: {max_length}")
        
        encoded = []
        lengths = []
        
        for seq in tqdm(sequences, desc="Encoding"):
            # Convert to uppercase and encode
            seq_upper = str(seq).upper()
            
            # Encode each amino acid
            encoded_seq = [self.aa_to_int.get(aa, self.aa_to_int['X']) for aa in seq_upper]
            
            # Truncate if too long
            if len(encoded_seq) > max_length:
                encoded_seq = encoded_seq[:max_length]
            
            # Store original length
            lengths.append(len(encoded_seq))
            
            # Pad if too short
            if len(encoded_seq) < max_length:
                encoded_seq = encoded_seq + [0] * (max_length - len(encoded_seq))
            
            encoded.append(encoded_seq)
        
        encoded_array = np.array(encoded, dtype=np.int32)
        lengths_array = np.array(lengths, dtype=np.int32)
        
        print(f"✓ Encoded shape: {encoded_array.shape}")
        print(f"✓ Avg sequence length: {np.mean(lengths):.1f}")
        print(f"✓ Length range: {np.min(lengths)} - {np.max(lengths)}")
        
        return encoded_array, lengths_array
    
    def prepare_go_labels(self, df, min_frequency=5):
        """
        Prepare multi-label GO annotations
        
        Args:
            df: DataFrame with GO columns
            min_frequency: Minimum term frequency (filter rare terms)
        
        Returns:
            Label matrix, MultiLabelBinarizer, GO term info
        """
        print(f"\n{'='*70}")
        print("PREPARING GO LABELS")
        print(f"{'='*70}")
        
        # Collect all GO annotations per protein
        go_lists = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing GO terms"):
            go_terms = set()
            
            # Extract GO terms from all columns
            for col in ['go_mf', 'go_bp', 'go_cc', 'go_all']:
                if col in df.columns and pd.notna(row[col]) and row[col]:
                    terms = str(row[col]).split(';')
                    go_terms.update([t.strip() for t in terms if t.strip()])
            
            go_lists.append(list(go_terms))
        
        print(f"✓ Collected GO annotations for {len(go_lists)} proteins")
        
        # Count term frequencies
        from collections import Counter
        term_counts = Counter()
        for terms in go_lists:
            term_counts.update(terms)
        
        print(f"✓ Total unique GO terms: {len(term_counts)}")
        
        # Filter rare terms
        if min_frequency > 1:
            frequent_terms = {term for term, count in term_counts.items() if count >= min_frequency}
            print(f"✓ Terms with frequency >= {min_frequency}: {len(frequent_terms)}")
            
            # Filter go_lists
            go_lists_filtered = [
                [term for term in terms if term in frequent_terms]
                for terms in go_lists
            ]
            
            # Remove proteins with no annotations after filtering
            valid_indices = [i for i, terms in enumerate(go_lists_filtered) if len(terms) > 0]
            print(f"✓ Proteins with valid annotations: {len(valid_indices)}")
            
            go_lists = [go_lists_filtered[i] for i in valid_indices]
        else:
            valid_indices = list(range(len(go_lists)))
        
        # Create multi-label binarizer
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(go_lists)
        
        print(f"\n✓ Label matrix shape: {y.shape}")
        print(f"✓ Number of GO classes: {len(mlb.classes_)}")
        print(f"✓ Average labels per protein: {y.sum(axis=1).mean():.2f}")
        print(f"✓ Label sparsity: {(1 - y.mean())*100:.2f}%")
        
        # Create GO term info
        go_info = {
            'classes': mlb.classes_.tolist(),
            'num_classes': len(mlb.classes_),
            'term_frequencies': {term: int(term_counts[term]) for term in mlb.classes_},
            'min_frequency': min_frequency
        }
        
        return y, mlb, go_info, valid_indices
    
    def process_split(self, split_name, max_length=1000, min_frequency=5):
        """
        Process a single data split (train/val/test)
        
        Args:
            split_name: 'train', 'val', or 'test'
            max_length: Maximum sequence length
            min_frequency: Minimum GO term frequency
        
        Returns:
            X (encoded sequences), y (labels), lengths
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'='*70}")
        
        # Load data
        file_path = self.raw_dir / f"{split_name}.csv"
        df = pd.read_csv(file_path)
        
        print(f"✓ Loaded {len(df)} proteins from {file_path}")
        
        # Encode sequences
        X, lengths = self.encode_sequences(df['sequence'].values, max_length)
        
        # Prepare labels
        if split_name == 'train':
            # Fit MultiLabelBinarizer on training data
            y, self.mlb, self.go_info, valid_indices = self.prepare_go_labels(df, min_frequency)
            
            # Save artifacts
            with open(self.processed_dir / 'mlb.pkl', 'wb') as f:
                pickle.dump(self.mlb, f)
            
            with open(self.processed_dir / 'go_info.json', 'w') as f:
                json.dump(self.go_info, f, indent=2)
            
            print(f"✓ Saved MultiLabelBinarizer and GO info")
        else:
            # Load pre-fitted MultiLabelBinarizer
            with open(self.processed_dir / 'mlb.pkl', 'rb') as f:
                self.mlb = pickle.load(f)
            
            # Transform using fitted binarizer
            go_lists = []
            for idx, row in df.iterrows():
                go_terms = set()
                for col in ['go_mf', 'go_bp', 'go_cc', 'go_all']:
                    if col in df.columns and pd.notna(row[col]) and row[col]:
                        terms = str(row[col]).split(';')
                        go_terms.update([t.strip() for t in terms if t.strip()])
                go_lists.append(list(go_terms))
            
            y = self.mlb.transform(go_lists)
            valid_indices = list(range(len(df)))
            
            print(f"✓ Transformed labels using fitted binarizer")
            print(f"✓ Label matrix shape: {y.shape}")
        
        # Filter by valid indices
        X = X[valid_indices]
        lengths = lengths[valid_indices]
        
        # Save processed data
        output_file = self.processed_dir / f"{split_name}_data.npz"
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            lengths=lengths,
            accessions=df.iloc[valid_indices]['accession'].values
        )
        
        print(f"\n✓ Saved processed data to: {output_file}")
        print(f"✓ Final dataset size: {len(X)} samples")
        
        return X, y, lengths
    
    def save_config(self, max_length=1000):
        """Save preprocessing configuration"""
        config = {
            'max_length': max_length,
            'vocab_size': len(self.aa_to_int),
            'aa_to_int': self.aa_to_int,
            'int_to_aa': self.int_to_aa,
            'amino_acids': self.amino_acids
        }
        
        config_file = self.processed_dir / 'preprocess_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Saved preprocessing config to: {config_file}")
    
    def get_dataset_statistics(self):
        """Calculate and display dataset statistics"""
        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print(f"{'='*70}")
        
        for split in ['train', 'val', 'test']:
            data_file = self.processed_dir / f"{split}_data.npz"
            if data_file.exists():
                data = np.load(data_file)
                X, y = data['X'], data['y']
                
                print(f"\n{split.upper()} SET:")
                print(f"  Samples: {len(X):,}")
                print(f"  Sequence shape: {X.shape}")
                print(f"  Label shape: {y.shape}")
                print(f"  Avg labels/protein: {y.sum(axis=1).mean():.2f}")
                print(f"  Memory size: {X.nbytes / (1024**2):.1f} MB")


def main():
    """Main execution"""
    print("\n" + "🧬 " * 35)
    print("PROTEOPREDICT - DATA PREPROCESSING MODULE")
    print("🧬 " * 35 + "\n")
    
    # Initialize preprocessor
    preprocessor = ProteoDataPreprocessor()
    
    # Configuration
    MAX_LENGTH = 1000  # Maximum sequence length
    MIN_FREQUENCY = 5  # Minimum GO term frequency
    
    print(f"Configuration:")
    print(f"  Max sequence length: {MAX_LENGTH}")
    print(f"  Min GO term frequency: {MIN_FREQUENCY}")
    
    # Process train set (fits MultiLabelBinarizer)
    X_train, y_train, lengths_train = preprocessor.process_split(
        'train', 
        max_length=MAX_LENGTH, 
        min_frequency=MIN_FREQUENCY
    )
    
    # Process validation set
    X_val, y_val, lengths_val = preprocessor.process_split(
        'val', 
        max_length=MAX_LENGTH
    )
    
    # Process test set
    X_test, y_test, lengths_test = preprocessor.process_split(
        'test', 
        max_length=MAX_LENGTH
    )
    
    # Save configuration
    preprocessor.save_config(max_length=MAX_LENGTH)
    
    # Display statistics
    preprocessor.get_dataset_statistics()
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nProcessed data saved in: {preprocessor.processed_dir}")
    print(f"\nFiles created:")
    print(f"  - train_data.npz ({len(X_train):,} samples)")
    print(f"  - val_data.npz ({len(X_val):,} samples)")
    print(f"  - test_data.npz ({len(X_test):,} samples)")
    print(f"  - mlb.pkl (MultiLabelBinarizer)")
    print(f"  - go_info.json (GO term information)")
    print(f"  - preprocess_config.json (Configuration)")
    print(f"\nNext step: Build and train model")
    print(f"  python -m src.proteopredict.training.train")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()