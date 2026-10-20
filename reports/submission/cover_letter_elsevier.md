# Cover Letter to the Editor-in-Chief

**To:** Prof. Yuan-Ting Zhang  
Editor-in-Chief, *Computers in Biology and Medicine* (Elsevier)  

**Submission Type:** Original Research Article  
**Manuscript Title:** ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across the Homology Twilight Zone  
**Authors:** Nikhil Dehariya  
**Affiliation:** Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India  
**Corresponding Author Email:** `nikhildehariya101@gmail.com`  
**Date:** October 20, 2026  

---

Dear Prof. Yuan-Ting Zhang and Editorial Board Members,

We are pleased to submit our original research manuscript titled **"ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across the Homology Twilight Zone"** for consideration as an **Original Research Article** in *Computers in Biology and Medicine*.

### 1. Scientific Significance & Biomedical Bottleneck
High-throughput automated prediction of Gene Ontology (GO) terms directly from primary amino acid sequences is crucial for precision medicine, structural biology, and targeted drug discovery. However, state-of-the-art homology transfer baselines (e.g., BLAST and $k$-NN transfer) suffer catastrophic degradation in the **"homology twilight zone"** ($< 30\%$ sequence identity), where divergence obscures evolutionary signals. Furthermore, while pre-trained protein language models (PLMs) capture global tertiary fold topology, standard pooled transformer embeddings frequently fail to preserve localized catalytic linear motifs and surface electrostatic properties essential for enzymatic specificity.

### 2. Methodological Innovation & Rigorous Benchmarks
In this study, we present **ProteoPredict**, a leak-free, tri-modal multi-branch deep architecture designed to resolve these limitations:
1. **Zero-Leakage Benchmark Curation:** Curated from 18,338 authentic Swiss-Prot reviewed human proteins (0 synthetic sequences) partitioned into 16,171 disjoint clusters via 3-mer TF-IDF nearest-neighbor graph clustering ($\text{cosine distance} \ge 0.50$, strictly $<30\%$ sequence identity). We enforce **strictly zero cluster overlap** between train (12,302), validation (2,691), and test (2,631) sets, eliminating deceptive data leakage.
2. **Tri-Modal Deep Network:** Simultaneously ingests:
   - Evolutionary Semantics: Meta AI ESM-2 (`esm2_t6_8M_UR50D`, 320D)
   - Alignment-Free Linear Motifs: 3-mer TF-IDF (1,500D)
   - Biophysical Profile: 430D (AAC 20D, DPC 400D, hydropathy, charge, pI, secondary structure 10D)
3. **Decoupled Heads & Focal Loss:** Features are routed through a shared trunk into independent sub-ontology decision heads (MF, BP, CC) optimized with multi-label Focal Loss ($\gamma = 2.0, \alpha = 0.35$).
4. **Empirical Superiority in the Twilight Zone:** On the unseen test set ($N=2,631$), ProteoPredict achieves **test Micro-$F_1$ ($F_{\max}$) = 0.5031** and **Macro-$F_1$ = 0.4997**. In the $<30\%$ identity regime, it outperforms Homology Transfer ($F_1 = 0.4607$) by capturing **$+35.5\%$ more true functional annotations** ($Recall = 0.5917$ vs. $0.4366$), achieving **$F_1 = 0.6146$** in Molecular Function.
5. **Lowest Semantic Distance:** Achieves CAFA semantic distance $S_{\min} = 19.86$ (Remaining Uncertainty $RU = 18.12$, Misinformation $MI = 8.13$), significantly outperforming the Naive Prior ($28.31$) and Homology Transfer ($22.76$).
6. **Active Site Explainability ($p = 0.00889$):** *In silico* Alanine-scanning perturbation on experimentally verified human active sites demonstrates that top-attended residues show a **3.09-fold statistical enrichment at verified catalytic residues** within the human EGFR Kinase domain (Mann-Whitney $U = 1168.5, p = 0.00889 < 0.01$).

### 3. Open Science & High-Throughput CPU Latency
ProteoPredict processes query sequences in **$0.14$ seconds on standard multi-core CPUs**. All source code, pre-trained checkpoints, reproduction suites, and an interactive Streamlit web application are freely available under the MIT License at `https://github.com/nikhildehariya/proteopredict`. A one-command script (`python scripts/reproduce_all.py`) verifies all empirical metrics in under 3 minutes.

### 4. Author Declarations
- This manuscript is original, has not been published previously, and is not currently under consideration by any other journal.
- The author has completed and approved the manuscript for submission.
- The author declares that there are no competing financial or non-financial interests.

### 5. Suggested Independent Reviewers
1. **Prof. Predrag Radivojac**, Khoury College of Computer Sciences, Northeastern University (`p.radivojac@northeastern.edu`). Lead organizer of the CAFA challenges; world authority on computational protein function evaluation.
2. **Dr. Maxat Kulmanov**, Computational Bioscience Research Center (CBRC), King Abdullah University of Science and Technology (`maxat.kulmanov@kaust.edu.sa`). Creator of DeepGOPlus and deep ontology predictors.
3. **Prof. Iddo Friedberg**, Department of Veterinary Microbiology & Pathology, Washington State University (`irokicki@gmail.com`). Co-organizer of CAFA and expert in automated annotation systems.
4. **Prof. Yana Bromberg**, Department of Biochemistry and Microbiology, Rutgers University (`yanab@rci.rutgers.edu`). Pioneer in sequence-function relationships and remote homology identification.

Thank you very much for your time and editorial consideration.

Sincerely,  
**Nikhil Dehariya**  
Department of Computer Science and Engineering (Artificial Intelligence and Data Science)  
Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India  
Email: `nikhildehariya101@gmail.com`
