import streamlit as st
import numpy as np
import joblib as jl
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load models & encoder ─────────────────────────────────────────────────────
# __file__ = .../kidney-failure-detection/app/app1.py
# Going one level up lands at the project root, then into outputs/models_10
BASE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'models_10')
)

@st.cache_resource
def load_artifacts():
    enc       = jl.load(os.path.join(BASE, 'encoding.pkl'))
    model_dt  = jl.load(os.path.join(BASE, 'model_dt.pkl'))
    model_gnb = jl.load(os.path.join(BASE, 'model_gnb.pkl'))
    return enc, model_dt, model_gnb

try:
    enc, model_dt, model_gnb = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ Could not load model files.\n\n**Path checked:** `{BASE}`\n\n**Error:** {e}")
    st.info("Make sure `encoding.pkl`, `model_dt.pkl`, and `model_gnb.pkl` exist in `outputs/models_10/`.")
    st.stop()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
}

.stApp { background: #0a0e1a; }

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0f1629 0%, #0a0e1a 100%);
    border-bottom: 1px solid #1e293b;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e293b;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* Result box */
.result-ckd {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-normal {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%       { box-shadow: 0 0 20px 4px rgba(239,68,68,0.2); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    50%       { box-shadow: 0 0 20px 4px rgba(34,197,94,0.2); }
}
.result-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-sub {
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    opacity: 0.7;
}

/* Engineered features badge */
.eng-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e293b;
    border: 1px solid #38bdf8;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    margin: 2px;
}

/* Streamlit overrides */
div[data-testid="stNumberInput"] > label,
div[data-testid="stSelectbox"] > label {
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="stNumberInput"] input {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    color: #0a0e1a;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem;
    cursor: pointer;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Model selector radio */
div[data-testid="stRadio"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
}

/* Probability bars */
.prob-bar-wrap { margin: 0.5rem 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
    margin-bottom: 4px;
}
.prob-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill-ckd {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f97316);
    transition: width 0.5s ease;
}
.prob-bar-fill-normal {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #10b981);
    transition: width 0.5s ease;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🩺 CKD Detector</h1>
  <p>Chronic Kidney Disease · Prediction Engine · models_10 · Feature-Engineered</p>
</div>
""", unsafe_allow_html=True)

# ── Engineered features info ───────────────────────────────────────────────────
st.markdown('<div class="section-label">⚙️ Feature Engineering Applied</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card" style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
  <span class="eng-badge">bp × sg → bp_sg</span>
  <span class="eng-badge">bgr × bu → bgr_bu</span>
  <span class="eng-badge">sc × sod → sc_sod</span>
  <span class="eng-badge">pot × hemo → pot_hemo</span>
  <span class="eng-badge">wc × rc → wc_rc</span>
  <span style="color:#64748b; font-size:0.75rem; font-family:'JetBrains Mono',monospace; margin-left:4px;">
    Original 14 features → 9 engineered features
  </span>
</div>
""", unsafe_allow_html=True)

# ── Model selector ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🤖 Model Selection</div>', unsafe_allow_html=True)
model_choice = st.radio(
    "Select classifier",
    ["Decision Tree", "Gaussian Naive Bayes"],
    horizontal=True,
    label_visibility="collapsed",
)
model = model_dt if model_choice == "Decision Tree" else model_gnb

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📋 Patient Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    age  = st.number_input("Age (years)",            min_value=1,     max_value=120,   value=45,    step=1)
    bp   = st.number_input("Blood Pressure (mm/Hg)", min_value=50.0,  max_value=200.0, value=80.0,  step=0.5)
    sg   = st.number_input("Specific Gravity",       min_value=1.000, max_value=1.030, value=1.020, step=0.001, format="%.3f")
    al   = st.number_input("Albumin (0–5)",          min_value=0,     max_value=5,     value=0,     step=1)
    su   = st.number_input("Sugar (0–5)",            min_value=0,     max_value=5,     value=0,     step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    bgr  = st.number_input("Blood Glucose Random (mg/dL)", min_value=50.0,  max_value=500.0, value=120.0, step=0.5)
    bu   = st.number_input("Blood Urea (mg/dL)",           min_value=1.0,   max_value=400.0, value=40.0,  step=0.5)
    sc   = st.number_input("Serum Creatinine (mg/dL)",     min_value=0.1,   max_value=20.0,  value=1.0,   step=0.1)
    sod  = st.number_input("Sodium (mEq/L)",               min_value=100.0, max_value=170.0, value=137.0, step=0.5)
    pot  = st.number_input("Potassium (mEq/L)",            min_value=2.0,   max_value=10.0,  value=4.5,   step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    hemo = st.number_input("Hemoglobin (g)",           min_value=3.0,  max_value=18.0,  value=14.0, step=0.1)
    pcv  = st.number_input("Packed Cell Volume (%)",   min_value=10.0, max_value=60.0,  value=41.0, step=0.5)
    wc   = st.number_input("WBC Count (cells/cumm)",   min_value=2000, max_value=26400, value=8000, step=100)
    rc   = st.number_input("RBC Count (millions/cmm)", min_value=1.0,  max_value=8.0,   value=5.0,  step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⚡ Run Prediction")

if predict_btn:
    # ── Feature engineering (must match training pipeline exactly) ────────────
    bp_sg    = bp  * sg
    bgr_bu   = bgr * bu
    sc_sod   = sc  * sod
    pot_hemo = pot * hemo
    wc_rc    = wc  * rc

    # Order: age, al, su, pcv, bp_sg, bgr_bu, sc_sod, pot_hemo, wc_rc
    features = np.array([[age, al, su, pcv, bp_sg, bgr_bu, sc_sod, pot_hemo, wc_rc]])

    # ── Encode ────────────────────────────────────────────────────────────────
    try:
        features_enc = enc.transform(features)
    except Exception:
        features_enc = features   # fallback: encoder not needed / passthrough

    # ── Predict ───────────────────────────────────────────────────────────────
    prediction = model.predict(features_enc)[0]
    label = "ckd" if prediction in [0, "ckd", "CKD"] else "notckd"

    # ── Probabilities (if model supports it) ──────────────────────────────────
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_enc)[0]

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🔬 Diagnosis Result</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        if label == "ckd":
            st.markdown("""
            <div class="result-ckd">
              <div class="result-title">⚠️ CKD Detected</div>
              <div class="result-sub">Chronic Kidney Disease · Consult a nephrologist</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
              <div class="result-title">✅ No CKD</div>
              <div class="result-sub">No Chronic Kidney Disease detected · Stay healthy!</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:#64748b;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;">
          Model Info
        </div>
        <div style="color:#94a3b8; font-size:0.82rem;">
          <span style="color:#38bdf8; font-family:'JetBrains Mono',monospace;">Model</span><br>
          {model_choice}<br><br>
          <span style="color:#38bdf8; font-family:'JetBrains Mono',monospace;">Features</span><br>
          9 (engineered)
        </div>
        """, unsafe_allow_html=True)

        if proba is not None:
            classes = model.classes_
            for i, cls in enumerate(classes):
                pct = proba[i] * 100
                cls_label  = "CKD"     if cls in [0, "ckd", "CKD"] else "Not CKD"
                bar_class  = "prob-bar-fill-ckd" if cls in [0, "ckd", "CKD"] else "prob-bar-fill-normal"
                st.markdown(f"""
                <div class="prob-bar-wrap">
                  <div class="prob-label"><span>{cls_label}</span><span>{pct:.1f}%</span></div>
                  <div class="prob-bar-bg">
                    <div class="{bar_class}" style="width:{pct}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Engineered feature breakdown ──────────────────────────────────────────
    with st.expander("🔧 View Engineered Feature Values"):
        ec1, ec2, ec3 = st.columns(3)
        vals = {
            "bp_sg":    bp_sg,
            "bgr_bu":   bgr_bu,
            "sc_sod":   sc_sod,
            "pot_hemo": pot_hemo,
            "wc_rc":    wc_rc,
            "age":      age,
            "al":       al,
            "su":       su,
            "pcv":      pcv,
        }
        for i, (k, v) in enumerate(vals.items()):
            col = [ec1, ec2, ec3][i % 3]
            col.markdown(f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px;
                        padding:0.8rem; margin:4px 0; text-align:center;">
              <div style="font-size:0.65rem; font-family:'JetBrains Mono',monospace;
                          color:#38bdf8; letter-spacing:2px; text-transform:uppercase;">{k}</div>
              <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0;
                          font-family:'JetBrains Mono',monospace;">{v:.3f}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#1e293b;
            font-size:0.7rem; font-family:'JetBrains Mono',monospace;">
  kidney-failure-detection · models_10 · UCI CKD Dataset
</div>
""", unsafe_allow_html=True)