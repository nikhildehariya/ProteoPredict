import os
import zipfile
import docx

def create_ieee_artifacts():
    # 1. IEEE Cover Letter Markdown
    cover_md = """# Cover Letter to the Editor-in-Chief

**To:** Prof. Tamer Kahveci  
Editor-in-Chief, *IEEE Transactions on Computational Biology and Bioinformatics (TCBB)*  
Department of Computer & Information Science & Engineering, University of Florida  

**Submission Type:** Regular Research Paper  
**Manuscript Title:** ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes  
**Authors:** Nikhil Dehariya (First & Corresponding Author), Prof. Ravinder Tanwar (Co-Author & Head of Department)  
**Affiliation:** Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India  
**Corresponding Author Email:** `nikhildehariya101@gmail.com`  
**Co-Author Institutional Email:** `ravindra.cse@jnctbhopal.ac.in`  
**Date:** October 25, 2026  

---

Dear Prof. Tamer Kahveci and Editorial Board Members,

We are pleased to submit our original research manuscript titled **"ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes"** for consideration as a **Regular Research Paper** in the *IEEE Transactions on Computational Biology and Bioinformatics (TCBB)*.

### 1. Scientific Significance & Computational Bottleneck
High-throughput automated prediction of Gene Ontology (GO) terms directly from primary amino acid sequences is a cornerstone of functional genomics, metabolic pathway reconstruction, and precision drug discovery. However, traditional homology transfer algorithms (e.g., BLAST and $k$-NN transfer) suffer marked performance collapse in low sequence and feature-space similarity regimes, where evolutionary divergence obscures simple alignment signals. Furthermore, while pre-trained protein language models (PLMs) capture global tertiary fold topology, standard pooled transformer embeddings frequently dilute localized catalytic linear motifs and surface biophysical properties essential for enzymatic specificity.

### 2. Methodological Innovation & Rigorous Benchmarks
In this study, we present **ProteoPredict**, a leak-free, tri-modal multi-branch deep architecture designed to resolve these computational challenges:
1. **Graph-Partitioned Non-Redundant Benchmark:** From an initial retrieval of 18,338 reviewed human Swiss-Prot proteins (18,288 unique), 17,624 fully annotated proteins were partitioned into 16,171 disjoint clusters via an alignment-free 3-mer TF-IDF nearest-neighbor graph heuristic ($d_{\cos} \le 0.50$, zero cluster overlap across 12,302 train / 2,691 val / 2,631 test splits), eliminating sequence identity leakage.
2. **Tri-Modal Deep Network:** Simultaneously ingests 2,250 dimensions across three complementary modalities:
   - Evolutionary Semantics: Meta AI ESM-2 (`esm2_t6_8M_UR50D`, 320D)
   - Alignment-Free Linear Motifs: 3-mer TF-IDF (1,500D)
   - Biophysical Profile: 430D (AAC 20D, DPC 400D, hydropathy, charge, pI, secondary structure 10D)
3. **Decoupled Heads & Focal Loss:** Features route through a shared trunk into independent sub-ontology decision heads (MF, BP, CC) optimized with multi-label Focal Loss ($\gamma = 2.0, \alpha = 0.35$).
4. **Empirical Performance Under Low Feature-Space Similarity:** On the held-out test cohort ($N=2,631$), ProteoPredict achieves **test Micro-$F_1$ = 0.5031** and **Macro-$F_1$ = 0.4997**. In this low 3-mer feature-space similarity regime (median similarity 0.306), it outperforms Nearest-Neighbor Transfer ($F_1 = 0.4607$) on the same test set by capturing a relative **$+35.5\%$ higher recall** ($0.5917$ vs. $0.4366$; $Precision = 0.4375$ vs. $0.4877$), reaching **$F_1 = 0.6146$** in Molecular Function.
5. **Lowest Semantic Distance:** Achieves CAFA semantic distance $S_{\min} = 19.86$ (Remaining Uncertainty $RU = 18.12$, Misinformation $MI = 8.13$), significantly outperforming the Naive Prior ($28.31$) and Nearest-Neighbor Transfer ($22.76$).
6. **Active Site Explainability:** Exploratory *in silico* sliding-window Alanine perturbation demonstrates a **1.31-fold functional sensitivity enrichment at verified catalytic residues** within the human EGFR Kinase domain (one-sided Mann-Whitney $U = 1168.5$, logged tie-corrected $p = 0.0091$; unadjusted asymptotic normal approximation $p = 0.017$), providing localized exploratory support for active-site sensitivity.

### 3. Open Science & Reproducibility
ProteoPredict processes single query sequences in **0.14 seconds for neural forward inference using cached ESM-2 embeddings**. All source code, pre-trained model checkpoints, and reproducible scripts are freely available at `https://github.com/nikhildehariya/proteopredict`. An interactive Streamlit graphical application and a standalone evaluation script (`scripts/evaluate_benchmark.py`) are provided for one-command end-to-end reproducibility.

### 4. Author Declarations
- This manuscript is original, has not been published previously, and is not currently under consideration by any other journal.
- All authors have reviewed and approved the manuscript for submission.
- The authors declare that there are no competing financial or non-financial interests.

### 5. Suggested Independent Reviewers
1. **Prof. Predrag Radivojac**, Khoury College of Computer Sciences, Northeastern University (`p.radivojac@northeastern.edu`). Lead organizer of the CAFA challenges; world authority on computational protein function evaluation.
2. **Dr. Maxat Kulmanov**, Computational Bioscience Research Center (CBRC), King Abdullah University of Science and Technology (`maxat.kulmanov@kaust.edu.sa`). Creator of DeepGOPlus and deep ontology predictors.
3. **Prof. Iddo Friedberg**, Department of Veterinary Microbiology & Pathology, Washington State University (`irokicki@gmail.com`). Co-organizer of CAFA and expert in automated annotation systems.
4. **Prof. Yana Bromberg**, Department of Biochemistry and Microbiology, Rutgers University (`yanab@rci.rutgers.edu`). Pioneer in sequence-function relationships and remote homology identification.

Thank you very much for your time and consideration.

Sincerely,  
**Nikhil Dehariya** (First & Corresponding Author)  
Department of Computer Science and Engineering (Artificial Intelligence and Data Science)  
Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India  
Email: `nikhildehariya101@gmail.com`  

**Prof. Ravinder Tanwar** (Co-Author & Head of Department)  
Department of Computer Science and Engineering  
Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India  
Email: `ravindra.cse@jnctbhopal.ac.in`
"""

    with open('reports/submission_ieee/cover_letter_tcbb.md', 'w', encoding='utf-8') as f:
        f.write(cover_md)

    # 2. IEEE Cover Letter Word Document
    doc = docx.Document()
    doc.add_heading('Cover Letter to the Editor-in-Chief', level=1)
    doc.add_paragraph('To: Prof. Tamer Kahveci, Editor-in-Chief, IEEE Transactions on Computational Biology and Bioinformatics (TCBB)')
    doc.add_paragraph('Department of Computer & Information Science & Engineering, University of Florida')
    doc.add_paragraph('Date: October 25, 2026')
    doc.add_paragraph('Submission Type: Regular Research Paper')
    doc.add_paragraph('Manuscript Title: ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc.add_paragraph('Authors: Nikhil Dehariya (First & Corresponding Author), Prof. Ravinder Tanwar (Co-Author & Head of Department)')
    doc.add_paragraph('Affiliation: Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India')
    doc.add_paragraph('Emails: nikhildehariya101@gmail.com, ravindra.cse@jnctbhopal.ac.in')
    doc.add_heading('Dear Prof. Tamer Kahveci and Editorial Board Members,', level=2)
    doc.add_paragraph('We are pleased to submit our original research manuscript titled "ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes" for consideration as a Regular Research Paper in IEEE Transactions on Computational Biology and Bioinformatics (TCBB).')
    doc.add_paragraph('High-throughput automated prediction of Gene Ontology (GO) terms directly from primary amino acid sequences is a cornerstone of functional genomics and precision drug discovery. However, traditional homology transfer algorithms suffer marked performance collapse in low sequence and feature-space similarity regimes where evolutionary divergence obscures simple alignment signals. Furthermore, while pre-trained protein language models (PLMs) capture global tertiary fold topology, standard pooled transformer embeddings frequently dilute localized catalytic linear motifs and surface biophysical properties essential for enzymatic specificity.')
    doc.add_paragraph('In this study, we present ProteoPredict, a leak-free, tri-modal multi-branch deep architecture resolving these limitations by integrating Meta AI ESM-2 (320D), 3-mer TF-IDF motifs (1,500D), and biophysical descriptors (430D) into decoupled sub-ontology heads optimized with multi-label Focal Loss.')
    doc.add_paragraph('On our non-redundant benchmark of 17,624 reviewed human proteins across 16,171 disjoint clusters (zero cluster overlap across 12,302 train, 2,691 val, and 2,631 test splits), ProteoPredict achieves test Micro-F1 = 0.5031 and Macro-F1 = 0.4997. In this low 3-mer feature-space similarity regime, it outperforms Nearest-Neighbor Transfer (F1 = 0.4607) on the same held-out test cohort by capturing a relative +35.5% higher recall (0.5917 vs. 0.4366) and reaching F1 = 0.6146 in Molecular Function. CAFA semantic distance is minimized to S_min = 19.86.')
    doc.add_paragraph('Furthermore, exploratory in silico sliding-window Alanine perturbation demonstrates a 1.31-fold functional sensitivity enrichment at verified catalytic residues within the human EGFR kinase domain (U = 1168.5, tie-corrected p = 0.0091; unadjusted normal approximation p = 0.017), providing localized exploratory support for active-site sensitivity.')
    doc.add_paragraph('This manuscript is original work, has not been published previously, and is not under consideration elsewhere. Both authors have reviewed and approved the manuscript. All code, trained weights, and benchmark splits are publicly hosted at https://github.com/nikhildehariya/proteopredict.')
    doc.add_paragraph('Sincerely,')
    doc.add_paragraph('Nikhil Dehariya\nLead Researcher & Corresponding Author\nJai Narain College of Technology (JNCT), Bhopal, India\nnikhildehariya101@gmail.com')
    doc.save('reports/submission_ieee/Cover_Letter_TCBB.docx')

    # 3. IEEE Supplementary Material Word Document
    doc_supp = docx.Document()
    doc_supp.add_heading('Supplementary Material: ProteoPredict', level=1)
    doc_supp.add_paragraph('Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc_supp.add_paragraph('Authors: Nikhil Dehariya, Prof. Ravinder Tanwar')
    doc_supp.add_paragraph('Affiliation: Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India')

    with open('reports/submission/supplementary_material.md', 'r', encoding='utf-8') as f:
        text = f.read()

    for para in text.split('\n\n'):
        p = para.strip()
        if not p:
            continue
        if p.startswith('# '):
            doc_supp.add_heading(p[2:], level=1)
        elif p.startswith('## '):
            doc_supp.add_heading(p[3:], level=2)
        elif p.startswith('### '):
            doc_supp.add_heading(p[4:], level=3)
        elif p.startswith('- '):
            for line in p.split('\n'):
                if line.startswith('- '):
                    doc_supp.add_paragraph(line[2:], style='List Bullet')
                else:
                    doc_supp.add_paragraph(line)
        else:
            doc_supp.add_paragraph(p)

    doc_supp.save('reports/submission_ieee/Supplementary_Material_TCBB.docx')

    # Copy markdown supplementary as well
    with open('reports/submission_ieee/supplementary_material.md', 'w', encoding='utf-8') as f:
        f.write(text)

    # 4. Create Overleaf-ready ZIP package
    zip_path = 'reports/submission_ieee/manuscript_ieee_overleaf.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('reports/submission_ieee/manuscript.tex', 'manuscript.tex')
        z.write('reports/submission_ieee/references.bib', 'references.bib')
        for root, dirs, files in os.walk('reports/submission_ieee/figures'):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, 'reports/submission_ieee')
                z.write(full_path, rel_path)

    print('Created IEEE artifacts successfully!')
    print('ZIP package created:', zip_path, 'Size:', os.path.getsize(zip_path), 'bytes')

if __name__ == '__main__':
    create_ieee_artifacts()
