"""
ProteoPredict Setup Configuration
"""

from setuptools import setup, find_packages

setup(
    name="proteopredict",
    version="1.0.0",
    description="AI-Powered Protein Function Prediction",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/proteopredict",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "tensorflow>=2.13.0",
        "biopython>=1.81",
        "scikit-learn>=1.3.0",
        "streamlit>=1.27.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)