import docx

def make_docs():
    # 1. Declaration of Interest Statement
    doc1 = docx.Document()
    doc1.add_heading('Declaration of Competing Interests', level=1)
    doc1.add_paragraph('Manuscript Title: ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc1.add_paragraph('Authors: Nikhil Dehariya (First & Corresponding Author), Prof. Ravinder Tanwar (Co-Author & Head of Department)')
    doc1.add_paragraph('Affiliation: Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India')
    doc1.add_heading('Declaration of Interest Statement:', level=2)
    doc1.add_paragraph('Both authors (Nikhil Dehariya and Prof. Ravinder Tanwar) have reviewed and confirmed that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.')
    doc1.add_paragraph('Conflict of Interest: None Declared.')
    doc1.save('reports/submission/Declaration_of_Interest.docx')
    print('Created reports/submission/Declaration_of_Interest.docx')

    # 2. Ethics Statement
    doc2 = docx.Document()
    doc2.add_heading('Ethics Approval and Consent to Participate', level=1)
    doc2.add_paragraph('Manuscript Title: ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc2.add_paragraph('Authors: Nikhil Dehariya, Prof. Ravinder Tanwar')
    doc2.add_paragraph('Affiliation: Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India')
    doc2.add_heading('Ethics Statement:', level=2)
    doc2.add_paragraph('Not applicable. This study utilizes publicly available, open-source computational biological databases (UniProtKB/Swiss-Prot and Gene Ontology) and does not involve human participants, animal experimentation, or biological tissue samples.')
    doc2.save('reports/submission/Ethics_Statement.docx')
    print('Created reports/submission/Ethics_Statement.docx')

    # 3. Highlights
    doc3 = docx.Document()
    doc3.add_heading('Research Highlights', level=1)
    doc3.add_paragraph('Manuscript Title: ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc3.add_paragraph('Journal: Computers in Biology and Medicine (Elsevier)')
    bullets = [
        'Tri-modal deep network predicts protein GO functions in low feature-space regimes.',
        'Non-redundant benchmark of 17,624 human proteins across 16,171 disjoint clusters.',
        'Achieves 0.5031 test Micro-F1, yielding +35.5% higher recall than transfer.',
        'Minimizes CAFA semantic distance to S_min = 19.86 with 0.6146 Molecular Function F1.',
        'In silico perturbation shows 1.31x sensitivity enrichment at catalytic sites.'
    ]
    for b in bullets:
        doc3.add_paragraph(b, style='List Bullet')
    doc3.save('reports/submission/Highlights.docx')
    print('Created reports/submission/Highlights.docx')

    # 4. Cover Letter
    doc4 = docx.Document()
    doc4.add_heading('Cover Letter to the Editor-in-Chief', level=1)
    doc4.add_paragraph('To: Prof. Yuan-Ting Zhang, Editor-in-Chief, Computers in Biology and Medicine (Elsevier)')
    doc4.add_paragraph('Date: October 23, 2026')
    doc4.add_paragraph('Submission Type: Full Length Article / Original Research Article')
    doc4.add_paragraph('Manuscript Title: ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc4.add_paragraph('Authors: Nikhil Dehariya (First & Corresponding Author), Prof. Ravinder Tanwar (Co-Author & Head of Department)')
    doc4.add_paragraph('Affiliation: Department of Computer Science and Engineering (Artificial Intelligence and Data Science), Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India')
    doc4.add_paragraph('Emails: nikhildehariya101@gmail.com, ravindra.cse@jnctbhopal.ac.in')
    doc4.add_heading('Dear Prof. Yuan-Ting Zhang and Editorial Board Members,', level=2)
    doc4.add_paragraph('We are pleased to submit our original research manuscript titled "ProteoPredict: Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes" for consideration as an Original Research Article in Computers in Biology and Medicine.')
    doc4.add_paragraph('High-throughput automated prediction of Gene Ontology (GO) terms directly from primary amino acid sequences is central to functional genomics and precision drug discovery. However, traditional homology transfer baselines suffer marked performance degradation in low feature-space similarity regimes where divergence obscures simple alignment signals. In this study, we present ProteoPredict, a leak-free, tri-modal multi-branch deep architecture resolving these limitations by integrating Meta AI ESM-2 (320D), 3-mer TF-IDF motifs (1,500D), and biophysical descriptors (430D) into decoupled sub-ontology heads optimized with multi-label Focal Loss.')
    doc4.add_paragraph('On our non-redundant benchmark of 17,624 reviewed human proteins across 16,171 disjoint clusters, ProteoPredict achieves test Micro-F1 = 0.5031 and Macro-F1 = 0.4997. In this low 3-mer feature-space similarity regime, it outperforms Nearest-Neighbor Transfer (F1 = 0.4607) on the same held-out test cohort by capturing a relative +35.5% higher recall (0.5917 vs. 0.4366) and reaching F1 = 0.6146 in Molecular Function. CAFA semantic distance is minimized to S_min = 19.86.')
    doc4.add_paragraph('Furthermore, exploratory in silico sliding-window Alanine perturbation demonstrates a 1.31-fold functional sensitivity enrichment at verified catalytic residues within the human EGFR kinase domain (U = 1168.5, tie-corrected p = 0.0091; unadjusted normal approximation p = 0.017), providing localized exploratory support for active-site sensitivity.')
    doc4.add_paragraph('This manuscript is original work, has not been published previously, and is not under consideration elsewhere. Both authors have reviewed and approved the manuscript. All code and benchmark splits are publicly hosted at https://github.com/nikhildehariya/proteopredict.')
    doc4.add_paragraph('Sincerely,')
    doc4.add_paragraph('Nikhil Dehariya\nLead Researcher & Corresponding Author\nJai Narain College of Technology (JNCT), Bhopal, India\nnikhildehariya101@gmail.com')
    doc4.save('reports/submission/Cover_Letter.docx')
    print('Created reports/submission/Cover_Letter.docx')

    # 5. Supplementary Material
    doc5 = docx.Document()
    doc5.add_heading('Supplementary Material: ProteoPredict', level=1)
    doc5.add_paragraph('Tri-Modal Deep Network Fusion for High-Resolution Protein Function Prediction Across Low 3-mer Feature-Space Similarity Regimes')
    doc5.add_paragraph('Authors: Nikhil Dehariya, Prof. Ravinder Tanwar')
    doc5.add_paragraph('Affiliation: Jai Narain College of Technology (JNCT), Bhopal, Madhya Pradesh, India')

    with open('reports/submission/supplementary_material.md', 'r', encoding='utf-8') as f:
        text = f.read()

    for para in text.split('\n\n'):
        p = para.strip()
        if not p:
            continue
        if p.startswith('# '):
            doc5.add_heading(p[2:], level=1)
        elif p.startswith('## '):
            doc5.add_heading(p[3:], level=2)
        elif p.startswith('### '):
            doc5.add_heading(p[4:], level=3)
        elif p.startswith('- '):
            for line in p.split('\n'):
                if line.startswith('- '):
                    doc5.add_paragraph(line[2:], style='List Bullet')
                else:
                    doc5.add_paragraph(line)
        else:
            doc5.add_paragraph(p)

    doc5.save('reports/submission/Supplementary_Material.docx')
    print('Created reports/submission/Supplementary_Material.docx')

if __name__ == '__main__':
    make_docs()
