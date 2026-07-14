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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;500;600;700&display=swap');
    
    html, body {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .medical-header {
        background: linear-gradient(135deg, #028090 0%, #002D62 100%);
        padding: 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(2, 128, 144, 0.15);
    }
    .medical-header h1 { color: white !important; font-weight: 700; font-size: 2.2rem !important; margin: 0; }
    .medical-header p { color: #E2E8F0; font-size: 1.05rem; margin-top: 0.5rem; margin-bottom: 0; }
    
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
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTENT -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
    st.markdown("## Clinical System Status")
    
    # System Operational Status - FIXED
    with st.container(border=True):
        st.markdown("🟢 **Inference Server:** Active")
        st.markdown("⚙️ **Backend:** TensorFlow CPU")
        st.markdown("🐍 **Python Env:** 3.13 Compatible")
    
    st.markdown("## Model Specification")
    with st.container(border=True):
        st.markdown("**Architecture:** DenseNet-121 (Fine-tuned)")
        st.markdown("**Dataset:** CheXpert v1.0-small")
        st.markdown("**Input Resolution:** 224x224 pixels")
    
    st.markdown("## Clinical Guidelines")
    st.info(
        "1. Ensure X-ray is in a frontal view.\n"
        "2. Check image orientation.\n"
        "3. Standard DICOM/JPG/PNG supported."
    )

# ----------------- MAIN TITLE HEADER -----------------
st.markdown("""
<div class="medical-header">
    <h1>🩺 CheXpert Chest X-Ray Diagnostic Assistant</h1>
    <p>AI-assisted multi-label screening and pathological classification tool.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- LAYOUT -----------------
col_left, col_right = st.columns([5, 7], gap="large")

# ----------------- FUNCTIONS -----------------
def draw_custom_progress_bar(label, prob):
    pct = prob * 100
    if prob >= 0.50: bar_color = "#E63946"
    elif prob >= 0.25: bar_color = "#F4A261"
    else: bar_color = "#2A9D8F"
    
    st.markdown(f"""
    <div style="margin-bottom: 0.95rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.3rem;">
            <span style="color: #334155;">{label}</span>
            <span style="color: {bar_color}; font-weight: 600;">{pct:.1f}%</span>
        </div>
        <div style="background-color: #E2E8F0; border-radius: 9999px; height: 8px; width: 100%; overflow: hidden;">
            <div style="background-color: {bar_color}; height: 100%; width: {pct}%; border-radius: 9999px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_prediction_chart(probs, labels):
    sorted_idx = np.argsort(probs)
    sorted_probs = [probs[i] for i in sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    colors = ['#E63946' if p >= 0.50 else '#F4A261' if p >= 0.25 else '#2A9D8F' for p in sorted_probs]
            
    fig = go.Figure(go.Bar(
        x=sorted_probs, y=sorted_labels, orientation='h',
        marker=dict(color=colors),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>'
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat='.0%', gridcolor='#E2E8F0'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10), height=320,
    )
    return fig

# ----------------- LEFT PANEL -----------------
with col_left:
    st.markdown('<div class="dashboard-card"><div class="card-title">📁 Upload Chest X-Ray</div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.markdown('<div class="dashboard-card"><div class="card-title">🔍 Image Preview</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- RIGHT PANEL -----------------
with col_right:
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            try:
                model = load_medical_model()
                preprocessed_img = preprocess_image(uploaded_file)
                preds = model.predict(preprocessed_img)[0]
            except Exception as e:
                st.error(f"Error: {e}")
                preds = None
        
        if preds is not None:
            st.markdown('<div class="dashboard-card"><div class="card-title">📊 Diagnostic Report</div>', unsafe_allow_html=True)
            tab_chart, tab_progress = st.tabs(["Chart", "Breakdown"])
            with tab_chart: st.plotly_chart(create_prediction_chart(preds, LABELS), use_container_width=True)
            with tab_progress:
                for label, prob in zip(LABELS, preds): draw_custom_progress_bar(label, prob)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical Alerts Section - CUSTOM STYLED HTML BOXES
            st.markdown('<div class="dashboard-card"><div class="card-title">⚠️ Clinical Alerts</div>', unsafe_allow_html=True)
            
            high_risk = [(l, p) for l, p in zip(LABELS, preds) if p >= 0.50]
            moderate_risk = [(l, p) for l, p in zip(LABELS, preds) if 0.25 <= p < 0.50]
            
            if high_risk:
                for l, p in high_risk: 
                    st.markdown(f"<div style='background-color: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; border-left: 5px solid #dc3545; margin-bottom: 10px; font-weight: 500;'>🚨 <b>High Risk:</b> {l} detected ({p*100:.1f}%). Urgent review recommended.</div>", unsafe_allow_html=True)
            if moderate_risk:
                for l, p in moderate_risk: 
                    st.markdown(f"<div style='background-color: #fff3cd; color: #856404; padding: 12px; border-radius: 8px; border-left: 5px solid #ffc107; margin-bottom: 10px; font-weight: 500;'>⚠️ <b>Monitor:</b> {l} indicates borderline probability ({p*100:.1f}%). Clinical correlation advised.</div>", unsafe_allow_html=True)
            if not high_risk and not moderate_risk:
                st.markdown("<div style='background-color: #d4edda; color: #155724; padding: 12px; border-radius: 8px; border-left: 5px solid #28a745; margin-bottom: 10px; font-weight: 500;'>✅ <b>No major findings.</b> Clinically correlate if patient is symptomatic.</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Disclaimer: AI screening aid only. Does not replace radiologist evaluation.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload an X-ray in the left panel to begin analysis.")