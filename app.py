import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils import load_medical_model, preprocess_image, LABELS

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CheXpert Chest X-Ray Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS THEME -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Typography & Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Hide Streamlit default components for a white-label premium feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Medical Header with blue/teal gradient */
    .medical-header {
        background: linear-gradient(135deg, #028090 0%, #002D62 100%);
        padding: 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(2, 128, 144, 0.15);
    }
    
    .medical-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.2rem !important;
        margin: 0;
    }
    
    .medical-header p {
        color: #E2E8F0;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Glassmorphism Cards */
    .dashboard-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1.25rem;
        color: #0F172A;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar styling enhancements */
    .sidebar-section {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTENT -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
    st.markdown("## Clinical System Status")
    
    # System Operational Status Card
    st.markdown("""
    <div class="sidebar-section">
        <span class="status-dot"></span><strong>Inference Server:</strong> Active<br>
        ⚙️ <strong>Backend:</strong> TensorFlow CPU<br>
        🐍 <strong>Python Env:</strong> 3.13 Compatible
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Model Specification")
    st.markdown("""
    <div class="sidebar-section" style="font-size: 0.9rem; line-height: 1.4;">
        <strong>Architecture:</strong> DenseNet-121 (Fine-tuned)<br>
        <strong>Training Dataset:</strong> CheXpert v1.0-small<br>
        <strong>Input Resolution:</strong> 224 &times; 224 pixels<br>
        <strong>Target Format:</strong> Multi-Label Classifier<br>
        <strong>Target Pathologies:</strong> 10 Classes
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Clinical Guidelines")
    st.info(
        "1. Ensure the uploaded chest X-ray is in a frontal view (PA or AP orientation).\n"
        "2. Check image orientation to ensure the cardiac silhouette is on the correct side.\n"
        "3. Review resolution: standard DICOM exports converted to JPEG/PNG are accepted."
    )

# ----------------- MAIN TITLE HEADER -----------------
st.markdown("""
<div class="medical-header">
    <h1>🩺 CheXpert Chest X-Ray Diagnostic Assistant</h1>
    <p>AI-assisted multi-label screening and pathological classification tool for front-line clinicians.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN COLUMNS LAYOUT -----------------
col_left, col_right = st.columns([5, 7], gap="large")

# ----------------- HELPER VISUAL FUNCTIONS -----------------
def draw_custom_progress_bar(label, prob):
    """
    Renders a custom styled medical progress bar.
    Color codes based on classification thresholds:
    - >= 50%: Red (Danger/Action Required)
    - >= 25% and < 50%: Yellow/Orange (Warning/Borderline)
    - < 25%: Teal (Low/Normal)
    """
    pct = prob * 100
    if prob >= 0.50:
        bar_color = "#E63946"  # Coral Red
    elif prob >= 0.25:
        bar_color = "#F4A261"  # Sandy Orange
    else:
        bar_color = "#2A9D8F"  # Medical Teal
        
    html = f"""
    <div style="margin-bottom: 0.95rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.3rem;">
            <span style="color: #334155;">{label}</span>
            <span style="color: {bar_color}; font-weight: 600;">{pct:.1f}%</span>
        </div>
        <div style="background-color: #E2E8F0; border-radius: 9999px; height: 8px; width: 100%; overflow: hidden;">
            <div style="background-color: {bar_color}; height: 100%; width: {pct}%; border-radius: 9999px; transition: width 0.5s ease-in-out;"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_prediction_chart(probs, labels):
    """
    Creates an interactive Plotly horizontal bar chart sorted by confidence.
    """
    # Sort for plot
    sorted_idx = np.argsort(probs)
    sorted_probs = [probs[i] for i in sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    
    # Map colors
    colors = []
    for p in sorted_probs:
        if p >= 0.50:
            colors.append('#E63946')
        elif p >= 0.25:
            colors.append('#F4A261')
        else:
            colors.append('#2A9D8F')
            
    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            title="Classification Confidence",
            range=[0, 1],
            tickformat='.0%',
            gridcolor='#E2E8F0',
            zerolinecolor='#E2E8F0'
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0)'
        ),
        font=dict(family='Outfit, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    return fig

# ----------------- LEFT PANEL: INPUT & UPLOAD -----------------
with col_left:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">📁 Upload Chest X-Ray</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select a chest X-ray image (PNG, JPG, or JPEG)...",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Image Preview</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        
        # Display image specs
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#64748B; margin-top: 0.5rem;">
            <strong>Filename:</strong> {uploaded_file.name}<br>
            <strong>Type:</strong> {uploaded_file.type}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: #E2E8F0; height: 350px; border-radius: 12px; display: flex; align-items: center; justify-content: center; border: 2px dashed #94A3B8; color: #64748B;">', unsafe_allow_html=True)
        st.write("Awaiting chest X-ray upload...")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- RIGHT PANEL: DIAGNOSTICS & ANALYSIS -----------------
with col_right:
    if uploaded_file is not None:
        # Load model and preprocess image
        with st.spinner("Analyzing image... (Loading CheXpert model and preprocessing X-ray)"):
            try:
                # Load pre-trained model
                model = load_medical_model()
                
                # Preprocess the input image
                preprocessed_img = preprocess_image(uploaded_file)
                
                # Perform prediction
                preds = model.predict(preprocessed_img)[0]
            except Exception as e:
                st.error(f"Failed to run diagnostics. Error: {e}")
                preds = None
        
        if preds is not None:
            # Main Diagnostic Report Card
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Multi-Label Diagnostic Report</div>', unsafe_allow_html=True)
            
            # Sub-columns to split Plotly and Progress bars
            tab_chart, tab_progress = st.tabs(["Interactive Analytics Chart", "Pathology Breakdown"])
            
            with tab_chart:
                fig = create_prediction_chart(preds, LABELS)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab_progress:
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                for label, prob in zip(LABELS, preds):
                    draw_custom_progress_bar(label, prob)
                    
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical Alerts Section
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚠️ Clinical Alerts & Action Items</div>', unsafe_allow_html=True)
            
            high_risk_findings = [(label, prob) for label, prob in zip(LABELS, preds) if prob >= 0.50]
            moderate_risk_findings = [(label, prob) for label, prob in zip(LABELS, preds) if 0.25 <= prob < 0.50]
            
            if high_risk_findings:
                for label, prob in high_risk_findings:
                    st.error(f"**Action Required - High Risk Finding:** {label} is detected with a classification confidence of **{prob*100:.1f}%**. Urgent clinical correlation and radiologist review is recommended.")
            
            if moderate_risk_findings:
                for label, prob in moderate_risk_findings:
                    st.warning(f"**Monitor/Correlate:** {label} indicates a borderline probability of **{prob*100:.1f}%**. Clinical checkup is recommended.")
            
            if not high_risk_findings and not moderate_risk_findings:
                st.success("✔️ **No major findings exceeded the screening threshold (25%).** The chest X-ray appears to have low-probability findings across all target classes. Clinically correlate if patient is symptomatic.")
            
            st.markdown("""
            <div style="font-size:0.8rem; background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 0.75rem; border-radius: 4px; color: #78350F; margin-top: 1rem;">
                <strong>Disclaimer:</strong> This diagnostic tool is powered by a machine learning model trained on the CheXpert dataset. It is intended to be used as a screening aid only. It does not replace a professional evaluation by a licensed radiologist.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        # Diagnostic placeholder card
        st.markdown('<div class="dashboard-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 4rem 2rem; color: #64748B;">', unsafe_allow_html=True)
        st.markdown('<img src="https://img.icons8.com/color/96/000000/x-ray.png" style="opacity: 0.65; margin-bottom: 1.5rem;" width="80">', unsafe_allow_html=True)
        st.markdown('<h3>Awaiting Diagnostics</h3>', unsafe_allow_html=True)
        st.write("Upload a patient's chest X-ray image in the left panel to execute the multi-label diagnostic screening. The analysis will outline 10 primary pathologies including Cardiomegaly, Pneumonia, and Pneumothorax.")
        st.markdown('</div>', unsafe_allow_html=True)