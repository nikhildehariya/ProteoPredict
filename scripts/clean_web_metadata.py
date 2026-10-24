import re

with open('reports/submission/manuscript.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract Title
m_title = re.search(r'\\title\{\\textbf\{\\Large (.*?)\}\}', text)
title = m_title.group(1).strip()

# Extract abstract
m_abs = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.DOTALL)
latex_abs = m_abs.group(1).strip()

# Clean LaTeX markup for web form (ASCII-safe for web portals)
clean_abs = latex_abs
clean_abs = clean_abs.replace(r'\textbf{ProteoPredict}', 'ProteoPredict')
clean_abs = clean_abs.replace(r'($320\text{D}$)', '(320D)')
clean_abs = clean_abs.replace(r'($1,500\text{D}$)', '(1,500D)')
clean_abs = clean_abs.replace(r'($430\text{D}$)', '(430D)')
clean_abs = clean_abs.replace(r'($\gamma = 2.0, \alpha = 0.35$)', '(gamma = 2.0, alpha = 0.35)')
clean_abs = clean_abs.replace(r'($d_{\cos} \le 0.50$, zero cluster overlap across $12,302$ train, $2,691$ val, and $2,631$ test splits)', '(d_cos <= 0.50, zero cluster overlap across 12,302 train, 2,691 val, and 2,631 test splits)')
clean_abs = clean_abs.replace(r'Micro-$F_1$', 'Micro-F1')
clean_abs = clean_abs.replace(r'Macro-$F_1$', 'Macro-F1')
clean_abs = clean_abs.replace(r'\textbf{0.5031}', '0.5031')
clean_abs = clean_abs.replace(r'\textbf{0.4997}', '0.4997')
clean_abs = clean_abs.replace(r'$3$-mer', '3-mer')
clean_abs = clean_abs.replace(r'$0.306$', '0.306')
clean_abs = clean_abs.replace(r'($F_1 = 0.4607$)', '(F1 = 0.4607)')
clean_abs = clean_abs.replace(r'\textbf{+35.5\% more true annotations}', '+35.5% more true annotations')
clean_abs = clean_abs.replace(r'(Recall $= 0.5917$ vs. $0.4366$, Micro-$F_1$ gain $+0.0424$)', '(Recall = 0.5917 vs. 0.4366, Micro-F1 gain +0.0424)')
clean_abs = clean_abs.replace(r'$F_1 = \textbf{0.6146}$', 'F1 = 0.6146')
clean_abs = clean_abs.replace(r'$S_{\min} = \textbf{19.86}$ (vs. $22.76$)', 'S_min = 19.86 (vs. 22.76)')
clean_abs = clean_abs.replace(r'\textit{In silico}', 'In silico')
clean_abs = clean_abs.replace(r'\textbf{1.31-fold functional sensitivity enrichment at verified catalytic residues}', '1.31-fold functional sensitivity enrichment at verified catalytic residues')
clean_abs = clean_abs.replace(r'Mann-Whitney $U = 1168.5$, logged tie-corrected $p = 0.0091 < 0.01$; unadjusted normal approximation $p = 0.017$', 'Mann-Whitney U = 1168.5, logged tie-corrected p = 0.0091 < 0.01; unadjusted normal approximation p = 0.017')
clean_abs = clean_abs.replace(r'\url{https://github.com/nikhildehariya/proteopredict}', 'https://github.com/nikhildehariya/proteopredict')
clean_abs = clean_abs.replace('$18,338$', '18,338').replace('$18,288$', '18,288').replace('$17,624$', '17,624').replace('$16,171$', '16,171')

# Extract Keywords
m_kw = re.search(r'\\noindent\\textbf\{Keywords:\}\s*(.*)', text)
keywords = m_kw.group(1).strip()

with open('reports/submission/validated_web_metadata.txt', 'w', encoding='utf-8') as out:
    out.write('=== TITLE ===\n' + title + '\n\n')
    out.write('=== ABSTRACT ===\n' + clean_abs + '\n\n')
    out.write(f'Word count: {len(clean_abs.split())} words\n\n')
    out.write('=== KEYWORDS ===\n' + keywords + '\n')

print('Successfully written validated_web_metadata.txt')
print(f'Word count: {len(clean_abs.split())} words')
