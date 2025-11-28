"""
ProteoPredict - Setup Verification Test
"""

import sys
from pathlib import Path

print("=" * 70)
print("PROTEOPREDICT - SETUP VERIFICATION")
print("=" * 70)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Test imports
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")

try:
    from Bio import SeqIO
    print(f"✓ BioPython imported successfully")
except ImportError as e:
    print(f"✗ BioPython import failed: {e}")

try:
    import streamlit as st
    print(f"✓ Streamlit imported successfully")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

# Check project structure
print("\n" + "=" * 70)
print("PROJECT STRUCTURE CHECK")
print("=" * 70)

required_dirs = [
    "data/raw",
    "data/processed",
    "src/proteopredict",
    "models",
    "results",
    "demo"
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"✓ {dir_path}")
    else:
        print(f"✗ {dir_path} - MISSING!")

print("\n" + "=" * 70)
print("🎉 PROTEOPREDICT SETUP VERIFICATION COMPLETE!")
print("=" * 70)
print("\nIf all tests passed, you're ready to start!")
print("Next: Run data download script")
print("=" * 70)