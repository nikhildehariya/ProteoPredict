import os
import re

with open('reports/submission/manuscript.tex', 'r', encoding='utf-8') as f:
    orig = f.read()

# Preamble for IEEE
preamble = r"""\documentclass[journal]{IEEEtran}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{cite}
\usepackage{url}
\usepackage{hyperref}
\usepackage{multirow}
\usepackage{adjustbox}
\usepackage{times}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\begin{document}

\title{ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes}

\author{Nikhil~Dehariya
        and~Ravinder~Tanwar%
\thanks{Manuscript submitted to IEEE Transactions on Computational Biology and Bioinformatics. Nikhil Dehariya (corresponding author) and Prof. Ravinder Tanwar are with the Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh 462038, India (e-mail: nikhildehariya101@gmail.com; ravindra.cse@jnctbhopal.ac.in).}%
}

\markboth{IEEE Transactions on Computational Biology and Bioinformatics}%
{Dehariya and Tanwar: ProteoPredict: Tri-Modal Deep Network Fusion Across Low 3-mer Feature-Space Similarity Regimes}

\maketitle

\begin{abstract}
Computational prediction of Gene Ontology (GO) terms from primary amino acid sequences is vital for functional genomics and drug discovery. Traditional homology transfer methods suffer marked performance degradation in low sequence similarity regimes, while monolithic protein language models dilute localized catalytic linear motifs and surface biophysical properties. We present \textbf{ProteoPredict}, a tri-modal multi-branch deep architecture integrating three complementary representations across 2,250 dimensions: Meta AI ESM-2 ($320\text{D}$) evolutionary embeddings, alignment-free $3$-mer TF-IDF ($1,500\text{D}$) motifs, and biophysical descriptors ($430\text{D}$) encoding charge, hydropathy, and secondary structure. Features route into decoupled sub-ontology heads optimized with multi-label Focal Loss ($\gamma = 2.0, \alpha = 0.35$). From $18,338$ retrieved human Swiss-Prot proteins ($18,288$ unique), $17,624$ fully annotated proteins were partitioned into $16,171$ disjoint clusters via a $3$-mer graph partition heuristic ($d_{\cos} \le 0.50$, zero cluster overlap across $12,302$ train, $2,691$ val, and $2,631$ test splits). Under CAFA protocols, ProteoPredict achieves a validation-calibrated test Micro-$F_1$ of \textbf{0.5031} and Macro-$F_1$ of \textbf{0.4997}. In this low $3$-mer feature-space similarity regime (median $3$-mer cosine similarity $0.306$), it outperforms homology transfer ($F_1 = 0.4607$) by capturing \textbf{$+35.5\%$ more true annotations} (Recall $= 0.5917$ vs. $0.4366$, Micro-$F_1$ gain $+0.0424$), reaching $F_1 = \textbf{0.6146}$ in Molecular Function and minimizing CAFA semantic distance to $S_{\min} = \textbf{19.86}$ (vs. $22.76$). \textit{In silico} sliding-window Alanine perturbation demonstrates a \textbf{1.31-fold functional sensitivity enrichment at verified catalytic residues} within the EGFR kinase domain (Mann-Whitney $U = 1168.5$, logged tie-corrected $p = 0.0091 < 0.01$; unadjusted normal approximation $p = 0.017$). Source code and benchmark splits are available at \url{https://github.com/nikhildehariya/proteopredict}.
\end{abstract}

\begin{IEEEkeywords}
Protein function prediction, Gene Ontology, Protein language models, Feature-space similarity, Deep learning, Active site explainability.
\end{IEEEkeywords}

"""

# Extract body from Section 1 to bibliography
start_pos = orig.find(r'\section{Introduction}')
end_pos = orig.find(r'{\small')
body = orig[start_pos:end_pos].strip()

# Adjust float specifiers: replace [H] with [!t]
body = body.replace(r'\begin{figure}[H]', r'\begin{figure}[!t]')
body = body.replace(r'\begin{figure}[t]', r'\begin{figure}[!t]')
body = body.replace(r'\begin{figure*}[t]', r'\begin{figure*}[!t]')

# Ending: Acknowledgment, Bibliography, Biographies
ending = r"""
\section*{Acknowledgment}
The authors thank the Department of Computer Science and Engineering (Artificial Intelligence and Data Science) at Jai Narain College of Technology (JNCT), Bhopal, for providing computational infrastructure and technical support. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

\bibliographystyle{IEEEtran}
\bibliography{references}

\begin{IEEEbiographynophoto}{Nikhil Dehariya}
received his B.Tech. degree in Computer Science and Engineering (Artificial Intelligence and Data Science) from Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India. His research interests focus on deep learning architectures, protein language models, multi-modal representation learning, and computational biology.
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Prof. Ravinder Tanwar}
is a Professor and Head of the Department of Computer Science and Engineering at Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India. His research areas include machine learning, deep learning applications, data engineering, and intelligent computing systems.
\end{IEEEbiographynophoto}

\end{document}
"""

full_ieee = preamble + body + ending

with open('reports/submission_ieee/manuscript.tex', 'w', encoding='utf-8') as f:
    f.write(full_ieee)

print('Generated reports/submission_ieee/manuscript.tex')
print('Length:', len(full_ieee), 'characters')
