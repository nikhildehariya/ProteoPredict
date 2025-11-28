# app_pro.py (FINAL ROBUST AND COMPLETE SCIENTIFIC VERSION)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
from pathlib import Path
from collections import Counter
import datetime
import re # Essential for Sequence Feature Mapping

# --- CRITICAL IMPORTS ---
try:
    # IMPORTANT: Ensure this function signature is correct in predict.py: 
    # def predict_go_terms(..., return_results=False, custom_threshold=None)
    from src.proteopredict.prediction.predict import predict_go_terms 
except ImportError:
    st.error("❌ CRITICAL ERROR: Could not import 'predict_go_terms'. Please check the path and environment setup ('src.proteopredict.prediction.predict').")
    st.stop() # Stop execution if critical import fails

# --- GO Term Mapping & Config (Including Simple Explanations) ---
GO_CATEGORY_MAP = {
    "GO:0000166": {"cat": "F", "desc": "Binds to a nucleotide molecule."}, "GO:0003677": {"cat": "F", "desc": "Binds specifically to DNA."}, 
    "GO:0003700": {"cat": "F", "desc": "Modulates gene transcription by binding DNA."}, "GO:0003723": {"cat": "F", "desc": "Binds specifically to RNA."}, 
    "GO:0004672": {"cat": "F", "desc": "Catalyzes phosphorylation of serine or threonine residues."}, "GO:0005515": {"cat": "F", "desc": "Non-covalent binding to another protein."}, 
    "GO:0005524": {"cat": "F", "desc": "Binds to ATP."}, "GO:0005576": {"cat": "C", "desc": "Region outside the outer surface of the plasma membrane."}, 
    "GO:0005634": {"cat": "C", "desc": "Enclosed by the nuclear envelope; contains the chromatin."}, "GO:0005654": {"cat": "C", "desc": "Part of the nucleus excluding the nucleolus."}, 
    "GO:0005737": {"cat": "C", "desc": "Substance of a cell within the cell membrane, excluding the nucleus."}, "GO:0005739": {"cat": "C", "desc": "Organelle that carries out cellular respiration."}, 
    "GO:0005783": {"cat": "C", "desc": "Network of membranes in the cytoplasm."}, "GO:0005886": {"cat": "C", "desc": "The membrane surrounding the cytoplasm."}, 
    "GO:0006351": {"cat": "P", "desc": "Synthesis of RNA from a DNA template."}, "GO:0006355": {"cat": "P", "desc": "Any process that modulates transcription."}, 
    "GO:0006412": {"cat": "P", "desc": "Synthesis of protein by ribosomes."}, "GO:0006468": {"cat": "P", "desc": "Addition of a phosphate group to a protein."}, 
    "GO:0006508": {"cat": "P", "desc": "Breakdown of proteins or peptides."}, "GO:0006810": {"cat": "P", "desc": "Movement of substances across a membrane."}, 
    "GO:0007049": {"cat": "P", "desc": "Process by which a cell duplicates its contents and divides."}, "GO:0007165": {"cat": "P", "desc": "Transmission of an extracellular signal into the cell."}, 
    "GO:0008270": {"cat": "F", "desc": "Binds to zinc ion."}, "GO:0008283": {"cat": "P", "desc": "Increase in cell numbers by cell division."}, 
    "GO:0016020": {"cat": "C", "desc": "A boundary separating the cell from its environment."}, "GO:0016740": {"cat": "F", "desc": "Catalyzes the transfer of a chemical group."}, 
    "GO:0046872": {"cat": "F", "desc": "Binds to any metal ion."}, "GO:0055114": {"cat": "F", "desc": "Catalyzes a reaction involving oxidation and reduction."}
}
CATEGORY_FULL_NAMES = {"P": "Biological Process", "F": "Molecular Function", "C": "Cellular Component", "All": "All Categories"}

# Hardcode file paths and metrics
MODEL_PATH = "models/weighted_run/best_model.h5"
DATA_DIR = "data/processed"
F1_SCORE = 0.4320
OPTIMAL_THRESHOLD_DEFAULT = 0.0100

# --- New Feature: Motif Checking Function ---
def check_motifs(sequence):
    """Checks the input sequence for critical functional motifs using regex."""
    
    motifs = {
        # Related to GO:0005524 (ATP binding) / Kinase activity
        "P-loop (ATP/GTP binding)": r"G[AG]X{4}G[KS]",  
        
        # Related to GO:0004672 (Serine/Threonine Kinase)
        "Kinase active site (Ser/Thr)": r"[IVL]H[LMV]D[LYF]K", 
        
        # Related to GO:0003677 (DNA binding) / GO:0008270 (Zinc ion binding)
        "C2H2 Zinc Finger": r"C.{2,4}C.{3}F.{5}L.{2}H.{3}H", 
        
        # Related to GO:0006508 (Proteolysis)
        "Aspartic Protease Signature": r"[FY][DNE]G[ST][ST]",
        
        # Related to GO:0046872 (Metal ion binding)
        "EF-hand (Calcium binding)": r"D.{2,4}D.{2,4}E", 
    }
    
    found_motifs = []
    
    # Search for all motifs
    for name, pattern in motifs.items():
        # Replace 'X' (any amino acid) with '.' (any character in regex)
        regex_pattern = pattern.replace('X', '.') 
        
        # Find all non-overlapping matches
        for match in re.finditer(regex_pattern, sequence):
            found_motifs.append({
                "Motif": name,
                "Pattern": pattern,
                "Start Position": match.start() + 1 # +1 for 1-based indexing
            })
            
    if not found_motifs:
        return None
        
    return pd.DataFrame(found_motifs)


# --- Report Generation Function (HTML Output for PDF) ---
def generate_report_html(sequence, source_id, df_results, custom_threshold):
    # Get clean sequence length and source ID from session state
    input_sequence_text = st.session_state.input_sequence.strip().upper()
    
    # Convert DataFrame to HTML table string
    df_report = df_results[['GO_ID', 'Category', 'Predicted_Function', 'Description', 'Confidence (%)']]
    results_table = df_report.to_html(
        index=False, 
        classes='report-table',
        justify='left'
    )
    
    report_html = f"""
    <html>
    <head>
        <title>ProteoPredict Scientific Annotation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #0d47a1; border-bottom: 3px solid #0d47a1; padding-bottom: 10px; font-size: 28px; }}
            h2 {{ color: #444; border-bottom: 1px solid #ccc; margin-top: 25px; padding-bottom: 5px; font-size: 20px; }}
            .param-table {{ width: 50%; margin-bottom: 20px; border-collapse: collapse; }}
            .param-table td {{ padding: 6px; border: 1px solid #eee; font-size: 14px; }}
            .param-table td:first-child {{ font-weight: bold; width: 40%; background-color: #f7f7f7; }}
            .sequence-box {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; white-space: pre-wrap; word-break: break-all; font-size: 12px; }}
            .report-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .report-table th, .report-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 12px; }}
            .report-table th {{ background-color: #e6e6e6; }}
            .caption {{ margin-top: 30px; font-size: 11px; color: #777; }}
        </style>
    </head>
    <body>
        <h1>ProteoPredict Scientific Annotation Report</h1>
        <p><strong>Generated On:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>1. Analysis Parameters</h2>
        <table class="param-table">
            <tr><td>Protein Source/ID</td><td>{source_id}</td></tr>
            <tr><td>Sequence Length</td><td>{len(input_sequence_text)} AAs</td></tr>
            <tr><td>Prediction Model</td><td>{Path(MODEL_PATH).name}</td></tr>
            <tr><td>Model F1-Score</td><td>{F1_SCORE:.4f}</td></tr>
            <tr><td>Applied Threshold</td><td>{custom_threshold:.4f} ({custom_threshold * 100:.2f}%)</td></tr>
        </table>

        <h2>Input Sequence (First 100 AAs)</h2>
        <div class="sequence-box">
        {input_sequence_text[:100]}...
        </div>

        <h2>2. Predicted Functional Annotations</h2>
        <p>The following Gene Ontology (GO) terms were predicted with a confidence score exceeding the set threshold.</p>
        
        {results_table}

        <h2>3. Conclusion</h2>
        <p>The model successfully annotated the protein sequence based on learned deep learning features, predicting key functions and cellular locations that may guide further experimental validation.</p>
        
        <p class="caption">© 2025 ProteoPredict Platform. Utilizing Deep Learning for Bioinformatics.</p>
    </body>
    </html>
    """
    return report_html


# --- Configuration & Styling ---
st.set_page_config(page_title="ProteoPredict: Scientific Report Generator", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .stButton>button { color: white; background-color: #0d47a1; border-radius: 8px; padding: 10px 24px; font-size: 18px; font-weight: bold; }
    h1 { color: #0d47a1; font-size: 3em; }
    h3 { color: #0d47a1; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Model Configuration (Threshold Slider) ---
st.sidebar.title("⚙️ Model Configuration")
if 'threshold' not in st.session_state: st.session_state.threshold = OPTIMAL_THRESHOLD_DEFAULT
custom_threshold = st.sidebar.slider(
    'Set Minimum Confidence Threshold (Adjust for Sensitivity)', min_value=0.00, max_value=0.50, value=st.session_state.threshold, step=0.01, format='%.2f'
)
st.sidebar.info(f"Active Threshold: **{custom_threshold:.2f}** ({custom_threshold * 100:.0f}%)")
st.sidebar.markdown("---")
st.sidebar.title("📊 System Metrics")
st.sidebar.markdown(f"**Core Metric (Micro F1):** **{F1_SCORE:.4f}**")

# --- Main App Interface and Input Logic ---
st.title("ProteoPredict 🧬")
st.header("Deep Learning Platform for Protein Functional Annotation")


st.markdown("### **1. Sequence Input & Parameters**")
col_seq, col_src, col_run = st.columns([2, 1, 1])

with col_seq:
    sequence = st.text_area(
        "Enter Protein Amino Acid Sequence:",
        value="MDKKYSIGLAIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRRQLPEKYAFIYDEVAKRRKNSLGYPVQITNSLVKVVADEALLSSDQIDLNADEVSQLKNRLRSTNKLGRALQLSGNEELKGKDGNGVTILAKEYAENLIIYVAKIDSETLIDEAKAGILEGTLERFMTNFY",
        height=150,
        key="input_sequence"
    )

with col_src:
    st.markdown("##### Source Tracking")
    source_id = st.text_input("Enter UniProt/PDB ID or Source Name", value="A0A175B5C1 (Cas9 - Bioinformatic Challenge)", key="source_id")

with col_run:
    st.markdown("##### Action")
    with st.container():
        st.write("") # Spacer
        if st.button("RUN PREDICTION ENGINE", type="primary", use_container_width=True):
            st.session_state['run_prediction'] = True
        else:
            st.session_state.setdefault('run_prediction', False)

# --- Prediction Logic Execution ---
if st.session_state.get('run_prediction'):
    
    # Robustly retrieve and clean sequence from session state
    clean_sequence = st.session_state.input_sequence.strip().upper()
    if clean_sequence.startswith('>'):
        clean_sequence = "".join(clean_sequence.split('\n')[1:])
    current_source_id = st.session_state.source_id

    if not clean_sequence or len(clean_sequence) < 10:
        st.error("❌ Invalid Input: Please enter a sequence of at least 10 amino acids.")
        st.session_state['run_prediction'] = False
        st.stop()
        
    if not Path(MODEL_PATH).exists():
        st.error(f"❌ CRITICAL ERROR: Model file '{MODEL_PATH}' is missing. Check your 'models' folder.")
        st.session_state['run_prediction'] = False
        st.stop()

    # 2. Run Model & Process Results
    with st.spinner(f"⏳ Annotating sequence from **{current_source_id}** using threshold **{custom_threshold:.2f}**..."):
        try:
            results = predict_go_terms(
                model_path=MODEL_PATH,
                data_dir=DATA_DIR,
                sequence=clean_sequence,
                return_results=True,
                custom_threshold=custom_threshold
            )

            if results:
                st.success("✅ Prediction Engine Complete. Analyzing Functional Domains...")
                
                # --- Create DataFrame & Add Features ---
                df = pd.DataFrame(results)
                df.columns = ["GO_ID", "Predicted_Function", "Confidence_Score"]
                df['Confidence_Score'] = df['Confidence_Score'].astype(float)
                
                df = df[df['Confidence_Score'] >= custom_threshold] 

                df['Category_Code'] = df['GO_ID'].map(lambda x: GO_CATEGORY_MAP.get(x, {}).get('cat', 'N/A'))
                df['Category'] = df['Category_Code'].map(CATEGORY_FULL_NAMES)
                df['Description'] = df['GO_ID'].map(lambda x: GO_CATEGORY_MAP.get(x, {}).get('desc', 'No detailed description available.'))
                df['Confidence (%)'] = (df['Confidence_Score'] * 100).round(2).astype(str) + '%'
                df['GO Link'] = df['GO_ID'].apply(lambda x: f"http://amigo.geneontology.org/amigo/term/{x}") 

                df = df.sort_values(by="Confidence_Score", ascending=False).reset_index(drop=True)
                
                # --- Results Display ---
                st.markdown("---")
                st.markdown(f"### **2. Annotation Results for: {current_source_id}**")

                # --- New Feature: Sequence Feature Mapping ---
                with st.expander("🔬 Sequence Feature Analysis (Motif and Composition)"):
                    
                    st.markdown(f"#### **Protein Length:** `{len(clean_sequence)}` Amino Acids")
                    
                    # 1. Motif Mapping
                    st.markdown("#### **Predicted Functional Motifs**")
                    motif_df = check_motifs(clean_sequence)
                    
                    if motif_df is not None:
                        st.success("✅ Critical functional motifs found! These support the GO term predictions.")
                        st.dataframe(
                            motif_df, 
                            hide_index=True, 
                            use_container_width=True, 
                            column_config={
                                "Start Position": st.column_config.NumberColumn(
                                    "Start Position",
                                    help="Starting position (1-based index) of the motif in the sequence."
                                )
                            }
                        )
                    else:
                        st.info("ℹ️ No common critical motifs (P-loop, Zinc Finger, etc.) found in the sequence.")
                        
                    # 2. Amino Acid Composition
                    st.markdown("#### **Amino Acid Composition**")
                    aa_counts = Counter(clean_sequence)
                    aa_df = pd.DataFrame(aa_counts.items(), columns=['AA', 'Count'])
                    aa_df['Percentage'] = (aa_df['Count'] / len(clean_sequence) * 100).round(1)
                    
                    col_plot, col_comp = st.columns([2, 1])
                    with col_plot:
                        fig_comp = px.bar(
                            aa_df.sort_values(by='Count', ascending=False).head(10), 
                            x='AA', y='Percentage', title='Top 10 Amino Acid Composition'
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    with col_comp:
                        st.dataframe(aa_df.sort_values(by='Percentage', ascending=False).set_index('AA'), use_container_width=True)
                
                # --- End of Sequence Feature Mapping ---

                # Filtering and Summary
                col_filt, col_summ = st.columns([1, 3])
                with col_filt:
                    category_filter = st.radio("Filter by GO Domain:", options=list(CATEGORY_FULL_NAMES.values()), index=3)
                
                df_display = df.copy()
                if category_filter != "All Categories":
                    df_display = df_display[df_display['Category'] == category_filter]
                
                with col_summ:
                    st.metric("Total Annotations Found", f"{len(df_display)} Terms")
                    
                
                # Final Table Display (Clickable Links and Description)
                cols_to_display = ["GO Link", "GO_ID", "Category", "Predicted_Function", "Description", "Confidence (%)"]
                
                st.dataframe(
                    df_display[cols_to_display].rename(columns={'GO_ID': 'GO ID', 'Predicted_Function': 'Predicted Function', 'Confidence (%)': 'Confidence'}),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "GO Link": st.column_config.LinkColumn(
                            "GO Link",
                            help="External link to AmiGO database for official term definition.",
                            display_text="AmiGO ↗️",
                            width="small"
                        ),
                        "Description": st.column_config.Column("Function Explanation", help="Brief scientific explanation of the predicted GO term.")
                    }
                )

                st.markdown("---")
                
                # --- Visualization and Reporting Section ---
                st.subheader("3. Visualization and Reporting")
                
                col_vis, col_report = st.columns([3, 1])

                with col_vis:
                    st.markdown("#### Confidence Heatmap Across Predicted Terms")
                    
                    # Heatmap Visualization
                    fig_heat = px.density_heatmap(
                        df, 
                        x="Category", 
                        y="Predicted_Function", 
                        z="Confidence_Score", 
                        histfunc="avg", 
                        color_continuous_scale="Plasma", 
                        title="GO Annotation Confidence Distribution"
                    )
                    fig_heat.update_layout(
                        height=500,
                        yaxis={'categoryorder':'total ascending', 'tickfont': {'size': 10}},
                        xaxis_title="",
                        coloraxis_colorbar=dict(title="Confidence Score")
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                
                with col_report:
                    # Generate the HTML report content
                    report_content = generate_report_html(clean_sequence, current_source_id, df, custom_threshold)

                    # Provide report for download as HTML (PDF Ready)
                    st.download_button(
                        label="⬇️ Download Scientific Report (.html)",
                        data=report_content,
                        file_name=f"ProteoPredict_Report_{current_source_id}_{datetime.datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    st.markdown("*(Open the .html file in your browser and use **Print > Save as PDF** for a professional report.)*")
                    
            else:
                st.warning(f"⚠️ No annotations found above the custom confidence threshold ({custom_threshold*100:.2f}%). Try lowering the threshold slider.")

        except Exception as e:
            st.error(f"❌ A critical error occurred during model execution. Check your `predict.py` file to ensure the function signature is correct: `def predict_go_terms(..., custom_threshold=None):`")
            st.exception(e)

st.markdown("---")
st.caption("© 2025 ProteoPredict Platform. Utilizing Deep Learning for Bioinformatics.")