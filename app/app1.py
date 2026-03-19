import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Sentinel — Kidney Disease Predictor",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  /* ── Root & Global ── */
  :root {
    --bg:         #0b0f1a;
    --bg2:        #111827;
    --bg3:        #1a2236;
    --border:     rgba(99,179,237,0.15);
    --accent:     #63b3ed;
    --accent2:    #4fd1c5;
    --danger:     #fc8181;
    --success:    #68d391;
    --warn:       #f6ad55;
    --text:       #e2e8f0;
    --muted:      #718096;
    --card-glow:  0 0 0 1px rgba(99,179,237,0.12), 0 8px 32px rgba(0,0,0,0.4);
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
  }

  [data-testid="stAppViewContainer"] > .main {
    background: var(--bg);
  }

  /* hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stDecoration"] { display:none; }

  /* ── Hero Header ── */
  .hero {
    background: linear-gradient(135deg, #0d1b2e 0%, #0b1a2f 40%, #0d2040 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 48px 36px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(79,209,197,0.06) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-tag {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent2);
    background: rgba(79,209,197,0.1);
    border: 1px solid rgba(79,209,197,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    margin-bottom: 14px;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    color: #f7fafc;
    margin: 0 0 10px;
    letter-spacing: -0.02em;
  }
  .hero-title span { color: var(--accent); }
  .hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    max-width: 560px;
  }
  .hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 28px;
    flex-wrap: wrap;
  }
  .stat-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: var(--muted);
  }
  .stat-chip strong { color: var(--text); font-weight: 500; }
  .stat-dot { width:8px; height:8px; border-radius:50%; background:var(--accent2); flex-shrink:0; }

  /* ── Section Headers ── */
  .sec-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
  }
  .sec-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
  }
  .sec-icon-blue  { background: rgba(99,179,237,0.12); border: 1px solid rgba(99,179,237,0.2); }
  .sec-icon-teal  { background: rgba(79,209,197,0.12); border: 1px solid rgba(79,209,197,0.2); }
  .sec-icon-red   { background: rgba(252,129,129,0.12); border: 1px solid rgba(252,129,129,0.2); }
  .sec-icon-yellow{ background: rgba(246,173,85,0.12);  border: 1px solid rgba(246,173,85,0.2); }
  .sec-icon-green { background: rgba(104,211,145,0.12); border: 1px solid rgba(104,211,145,0.2); }
  .sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.01em;
  }
  .sec-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }

  /* ── Cards / input wrappers ── */
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: var(--card-glow);
  }

  /* ── Streamlit widget overrides ── */
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  [data-testid="stNumberInput"] input:focus,
  [data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.15) !important;
  }
  [data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }

  /* Slider */
  [data-testid="stSlider"] { padding: 4px 0; }

  /* ── Submit Button ── */
  [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(49,130,206,0.35) !important;
  }
  [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(49,130,206,0.5) !important;
  }

  /* ── Result Banners ── */
  .result-ckd {
    background: linear-gradient(135deg, rgba(252,129,129,0.12), rgba(197,48,48,0.08));
    border: 1.5px solid rgba(252,129,129,0.4);
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin: 12px 0;
  }
  .result-ckd .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--danger);
    margin-bottom: 6px;
  }
  .result-notckd {
    background: linear-gradient(135deg, rgba(104,211,145,0.12), rgba(47,133,90,0.08));
    border: 1.5px solid rgba(104,211,145,0.4);
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin: 12px 0;
  }
  .result-notckd .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--success);
    margin-bottom: 6px;
  }
  .result-sub {
    font-size: 0.9rem;
    color: var(--muted);
    font-weight: 300;
  }

  /* ── Metric Cards ── */
  .metric-row { display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }
  .metric-card {
    flex:1; min-width:140px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
  }
  .metric-label { font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
  .metric-value { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; }
  .metric-val-danger { color: var(--danger); }
  .metric-val-success { color: var(--success); }
  .metric-val-accent  { color: var(--accent); }

  /* ── Info banner ── */
  .info-banner {
    display:flex; align-items:flex-start; gap:12px;
    background:rgba(99,179,237,0.06);
    border:1px solid rgba(99,179,237,0.2);
    border-radius:12px;
    padding:14px 18px;
    font-size:0.85rem;
    color:var(--muted);
    margin-top:24px;
  }
  .info-banner strong { color: var(--accent); }

  /* ── Reference table ── */
  .ref-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
  .ref-table th {
    text-align:left; padding:8px 12px;
    color:var(--muted); font-weight:500;
    border-bottom:1px solid var(--border);
    font-family:'Syne',sans-serif; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;
  }
  .ref-table td { padding:8px 12px; border-bottom:1px solid rgba(99,179,237,0.06); color:var(--text); }
  .ref-table tr:last-child td { border-bottom: none; }
  .badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:600;
  }
  .badge-green { background:rgba(104,211,145,0.15); color:var(--success); }
  .badge-red   { background:rgba(252,129,129,0.15); color:var(--danger); }
  .badge-blue  { background:rgba(99,179,237,0.15);  color:var(--accent); }

  /* ── Progress ring placeholder ── */
  .risk-gauge-wrap { display:flex; justify-content:center; padding:8px 0 16px; }

  /* labels */
  label, [data-testid="stWidgetLabel"] p {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    margin-bottom: 4px !important;
  }

  /* Expander */
  [data-testid="stExpander"] summary {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
  }
  [data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
  }

  hr { border-color: var(--border) !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(base_dir, '..', 'outputs', 'models', 'model_gnb.pkl')
    encoder_path = os.path.join(base_dir, '..', 'outputs', 'models', 'encoding.pkl')
    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

try:
    model, label_encoder = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(f"❌ Could not load model files: {e}\n\nMake sure `model_gnb.pkl` and `encoding.pkl` exist in `outputs/models/`.")

FEATURE_COLS = ['age','bp','sg','al','su','rbc','pc','pcc','ba',
                'bgr','bu','sc','sod','pot','hemo','pcv','wc','rc',
                'htn','dm','cad','appet','pe','ane']

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">🩺 AI Clinical Decision Support</div>
  <div class="hero-title">CKD <span>Sentinel</span></div>
  <div class="hero-sub">
    Advanced Chronic Kidney Disease risk assessment powered by Gaussian Naïve Bayes.
    Enter patient values across 24 biomarkers for an instant clinical prediction.
  </div>
  <div class="hero-stats">
    <div class="stat-chip"><div class="stat-dot"></div><strong>24</strong>&nbsp;Biomarkers Analysed</div>
    <div class="stat-chip"><div class="stat-dot" style="background:var(--warn)"></div><strong>GNB</strong>&nbsp;Model</div>
    <div class="stat-chip"><div class="stat-dot" style="background:var(--success)"></div><strong>Research Use</strong>&nbsp;Only</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Reference ranges sidebar card ─────────────────────────────────────────────
with st.expander("📖 Normal Reference Ranges — Quick Guide"):
    ref_data = {
        "Biomarker": ["Blood Pressure","Blood Glucose Random","Blood Urea","Serum Creatinine",
                      "Haemoglobin","Packed Cell Volume","WBC Count","RBC Count","Sodium","Potassium"],
        "Normal Range": ["60–80 mm/Hg","70–140 mg/dl","7–20 mg/dl","0.6–1.2 mg/dl",
                         "12–17 g/dl","36–50%","4000–11000","4.5–6.0 M/cmm","135–145 mEq/L","3.5–5.0 mEq/L"],
        "Status": ["normal","normal","normal","normal","normal","normal","normal","normal","optional","optional"],
    }
    ref_df = pd.DataFrame(ref_data)
    html_rows = ""
    for _, row in ref_df.iterrows():
        badge_cls = "badge-green" if row["Status"]=="normal" else "badge-blue"
        html_rows += f"""<tr>
          <td>{row['Biomarker']}</td>
          <td>{row['Normal Range']}</td>
          <td><span class="badge {badge_cls}">{row['Status']}</span></td>
        </tr>"""
    st.markdown(f"""
    <table class="ref-table">
      <thead><tr><th>Biomarker</th><th>Normal Range</th><th>Status</th></tr></thead>
      <tbody>{html_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ── FORM ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header">
  <div class="sec-icon sec-icon-blue">👤</div>
  <span class="sec-title">Patient Data Entry</span>
  <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

with st.form("predict_form"):

    # ── Demographics & Vitals ──────────────────────────────────────────────────
    st.markdown("""<div class="sec-header" style="margin-top:4px;">
      <div class="sec-icon sec-icon-blue">🏥</div>
      <span class="sec-title">Demographics & Vitals</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    age = c1.slider("Age (years)", min_value=1, max_value=100, value=48,
                    help="Patient's age in years")
    bp  = c2.slider("Blood Pressure (mm/Hg)", min_value=50, max_value=180, value=80,
                    help="Diastolic blood pressure in mm/Hg")

    # ── Urine Analysis ─────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-header">
      <div class="sec-icon sec-icon-teal">🧪</div>
      <span class="sec-title">Urine Analysis</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    sg  = c1.select_slider("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025], value=1.020)
    al  = c2.select_slider("Albumin (0–5)", options=[0,1,2,3,4,5], value=1)
    su  = c3.select_slider("Sugar (0–5)", options=[0,1,2,3,4,5], value=0)

    c1, c2, c3, c4 = st.columns(4)
    rbc = c1.selectbox("Red Blood Cells", ["normal","abnormal"])
    pc  = c2.selectbox("Pus Cell", ["normal","abnormal"])
    pcc = c3.selectbox("Pus Cell Clumps", ["notpresent","present"])
    ba  = c4.selectbox("Bacteria", ["notpresent","present"])

    # ── Blood Tests ────────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-header">
      <div class="sec-icon sec-icon-red">🩸</div>
      <span class="sec-title">Blood Tests</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    bgr  = c1.number_input("Blood Glucose Random (mg/dl)", 50, 500, 121)
    bu   = c2.number_input("Blood Urea (mg/dl)", 1, 400, 36)
    sc   = c3.number_input("Serum Creatinine (mg/dl)", 0.1, 40.0, 1.2, step=0.1, format="%.1f")
    hemo = c4.number_input("Haemoglobin (g/dl)", 1.0, 20.0, 15.4, step=0.1, format="%.1f")

    c1, c2, c3, _ = st.columns(4)
    pcv  = c1.number_input("Packed Cell Volume (%)", 10, 60, 44)
    wc   = c2.number_input("WBC Count (cells/cumm)", 2000, 26400, 7800)
    rc   = c3.number_input("RBC Count (M/cmm)", 0.0, 8.0, 5.2, step=0.1, format="%.1f")

    # ── Electrolytes ───────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-header">
      <div class="sec-icon sec-icon-yellow">🧂</div>
      <span class="sec-title">Electrolytes <span style="font-size:0.8em;color:#718096;font-weight:400">(optional)</span></span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    sod_input = c1.text_input("Sodium (mEq/L)", value="", placeholder="Leave blank if unknown")
    pot_input = c2.text_input("Potassium (mEq/L)", value="", placeholder="Leave blank if unknown")

    # ── Medical History ────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-header">
      <div class="sec-icon sec-icon-green">📋</div>
      <span class="sec-title">Medical History</span>
    </div>""", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    htn   = col_a.radio("Hypertension",        ["yes","no"], horizontal=True)
    dm    = col_b.radio("Diabetes Mellitus",   ["yes","no"], horizontal=True)
    cad   = col_c.radio("Coronary Artery Disease", ["no","yes"], horizontal=True)

    col_a, col_b, col_c = st.columns(3)
    appet = col_a.radio("Appetite",    ["good","poor"], horizontal=True)
    pe    = col_b.radio("Pedal Edema", ["no","yes"],   horizontal=True)
    ane   = col_c.radio("Anaemia",     ["no","yes"],   horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ Run Prediction Analysis", use_container_width=True, type="primary")


# ── RESULT ─────────────────────────────────────────────────────────────────────
if submitted and model_loaded:
    sod = float(sod_input) if sod_input.strip() else np.nan
    pot = float(pot_input) if pot_input.strip() else np.nan

    raw = {
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
        'bgr': bgr, 'bu': bu, 'sc': sc,
        'sod': sod, 'pot': pot,
        'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
        'htn': htn, 'dm': dm, 'cad': cad,
        'appet': appet, 'pe': pe, 'ane': ane,
    }

    df_input = pd.DataFrame([raw])
    df_transformed = label_encoder.transform(df_input)

    pred  = model.predict(df_transformed)[0]
    proba = model.predict_proba(df_transformed)[0] if hasattr(model, "predict_proba") else None

    LABEL_MAP = {0: "notckd", 1: "ckd", "ckd": "ckd", "notckd": "notckd"}
    label  = LABEL_MAP.get(str(pred).lower(), str(pred))
    is_ckd = label == "ckd"

    if hasattr(model, "classes_"):
        classes = [str(c).lower() for c in model.classes_]
    else:
        classes = ["notckd","ckd"]

    ckd_prob    = proba[classes.index("ckd")]    * 100 if proba is not None and "ckd"    in classes else (80 if is_ckd else 20)
    notckd_prob = proba[classes.index("notckd")] * 100 if proba is not None and "notckd" in classes else (100 - ckd_prob)

    st.markdown("---")
    st.markdown("""<div class="sec-header">
      <div class="sec-icon sec-icon-blue">📊</div>
      <span class="sec-title">Prediction Results</span>
      <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)

    # Result banner
    if is_ckd:
        st.markdown(f"""
        <div class="result-ckd">
          <div style="font-size:2.2rem;margin-bottom:8px;">⚠️</div>
          <div class="result-label">Chronic Kidney Disease Detected</div>
          <div class="result-sub">The model predicts a high likelihood of CKD based on the provided biomarkers.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-notckd">
          <div style="font-size:2.2rem;margin-bottom:8px;">✅</div>
          <div class="result-label">No CKD Detected</div>
          <div class="result-sub">The model does not find strong indicators of Chronic Kidney Disease.</div>
        </div>""", unsafe_allow_html=True)

    # ── Gauge + Donut ──────────────────────────────────────────────────────────
    col_gauge, col_donut = st.columns([3, 2])

    with col_gauge:
        st.markdown("##### 🎯 CKD Risk Score")
        gauge_color = "#fc8181" if is_ckd else "#68d391"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(ckd_prob, 1),
            delta={'reference': 50, 'increasing': {'color': '#fc8181'}, 'decreasing': {'color': '#68d391'}},
            number={'suffix': "%", 'font': {'size': 48, 'color': gauge_color, 'family': 'Syne'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#4a5568', 'tickfont': {'color': '#718096', 'size': 11}},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': '#1a2236',
                'bordercolor': '#2d3748',
                'steps': [
                    {'range': [0, 30],  'color': 'rgba(104,211,145,0.10)'},
                    {'range': [30, 60], 'color': 'rgba(246,173,85,0.10)'},
                    {'range': [60, 100],'color': 'rgba(252,129,129,0.10)'},
                ],
                'threshold': {
                    'line': {'color': '#f6ad55', 'width': 3},
                    'thickness': 0.8,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=10, l=20, r=20),
            height=260,
            font={'color': '#e2e8f0'},
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_donut:
        st.markdown("##### 📊 Class Probability")
        fig_donut = go.Figure(go.Pie(
            labels=["CKD", "Not CKD"],
            values=[ckd_prob, notckd_prob],
            hole=0.62,
            marker=dict(
                colors=["#fc8181", "#68d391"],
                line=dict(color='#0b0f1a', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color='#e2e8f0', family='DM Sans'),
            hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        ))
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=10, l=10, r=10),
            height=260,
            showlegend=True,
            legend=dict(
                font=dict(color='#a0aec0', size=11, family='DM Sans'),
                bgcolor='rgba(0,0,0,0)',
                orientation='h',
                yanchor='bottom', y=-0.1, xanchor='center', x=0.5
            ),
            annotations=[dict(
                text=f"<b>{round(ckd_prob,1)}%</b><br><span style='font-size:10'>CKD Risk</span>",
                x=0.5, y=0.5, font_size=16, showarrow=False,
                font=dict(color=gauge_color, family='Syne')
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    # ── Biomarker Radar ────────────────────────────────────────────────────────
    st.markdown("##### 🕸️ Patient Biomarker Profile (Normalised)")

    # Normalise key numeric values to 0–1 for radar
    radar_features = {
        'Blood\nPressure': min(bp / 120, 1.5),
        'Blood\nGlucose':  min(bgr / 200, 1.5),
        'Blood\nUrea':     min(bu / 80, 1.5),
        'Creatinine':      min(sc / 5, 1.5),
        'Haemoglobin':     min(hemo / 17, 1.0),
        'Albumin':         min(al / 5, 1.0),
        'Sugar':           min(su / 5, 1.0),
        'WBC Count':       min(wc / 12000, 1.5),
        'RBC Count':       min(rc / 6, 1.0),
        'PCV':             min(pcv / 50, 1.0),
    }
    cats   = list(radar_features.keys())
    vals   = list(radar_features.values())
    normal_vals = [0.65, 0.55, 0.3, 0.2, 0.88, 0.2, 0.0, 0.6, 0.85, 0.85]

    cats_closed   = cats + [cats[0]]
    vals_closed   = vals + [vals[0]]
    normal_closed = normal_vals + [normal_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normal_closed, theta=cats_closed, fill='toself',
        name='Normal Range',
        line=dict(color='#4fd1c5', width=1.5, dash='dot'),
        fillcolor='rgba(79,209,197,0.06)',
        marker=dict(size=0),
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill='toself',
        name='Patient Values',
        line=dict(color='#63b3ed', width=2.5),
        fillcolor=f'rgba({"252,129,129" if is_ckd else "104,211,145"},0.15)',
        marker=dict(size=5, color='#63b3ed'),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(26,34,54,0.6)',
            radialaxis=dict(visible=True, range=[0,1.5], gridcolor='rgba(99,179,237,0.12)',
                            tickfont=dict(color='#718096', size=9), showticklabels=False),
            angularaxis=dict(gridcolor='rgba(99,179,237,0.12)',
                             tickfont=dict(color='#a0aec0', size=10, family='DM Sans')),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=40, r=40),
        height=360,
        showlegend=True,
        legend=dict(
            font=dict(color='#a0aec0', size=11, family='DM Sans'),
            bgcolor='rgba(17,24,39,0.8)',
            bordercolor='rgba(99,179,237,0.2)',
            borderwidth=1,
            orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5
        )
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # ── Feature Importance ────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        st.markdown("##### 🔬 Top Contributing Features")
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(12)

        fig_bar = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation='h',
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,'#2b6cb0'],[0.5,'#63b3ed'],[1,'#4fd1c5']],
                showscale=False,
                line=dict(width=0)
            ),
            hovertemplate='%{y}: %{x:.4f}<extra></extra>',
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition='outside',
            textfont=dict(color='#718096', size=10),
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.08)',
                       zeroline=False, tickfont=dict(color='#718096', size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(color='#a0aec0', size=11, family='DM Sans')),
            margin=dict(t=10, b=10, l=10, r=60),
            height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # ── Numeric biomarker bar chart ────────────────────────────────────────────
    st.markdown("##### 📈 Key Blood Marker Comparison vs Normal")
    markers      = ['Blood Pressure','Glucose','Urea','Creatinine','Haemoglobin','PCV','WBC','RBC']
    patient_vals = [bp,             bgr,       bu,    sc,          hemo,          pcv,  wc/1000, rc]
    normal_mid   = [80,             100,       15,    1.0,         14.5,          43,   8.0,     5.0]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name='Normal Midpoint',
        x=markers, y=normal_mid,
        marker_color='rgba(79,209,197,0.4)',
        marker_line=dict(color='rgba(79,209,197,0.8)', width=1.5),
        hovertemplate='%{x}<br>Normal: %{y}<extra></extra>',
    ))
    fig_compare.add_trace(go.Bar(
        name='Patient Value',
        x=markers, y=patient_vals,
        marker_color=[
            '#fc8181' if abs(p - n) / max(n,0.1) > 0.3 else '#63b3ed'
            for p, n in zip(patient_vals, normal_mid)
        ],
        hovertemplate='%{x}<br>Patient: %{y}<extra></extra>',
    ))
    fig_compare.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickfont=dict(color='#a0aec0', size=10, family='DM Sans')),
        yaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.08)',
                   tickfont=dict(color='#718096', size=10), zeroline=False),
        margin=dict(t=10, b=10),
        height=300,
        legend=dict(font=dict(color='#a0aec0', size=11), bgcolor='rgba(0,0,0,0)',
                    orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        bargap=0.25, bargroupgap=0.05,
    )
    st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})

    # ── Input Summary Table ────────────────────────────────────────────────────
    with st.expander("📋 Full Input Summary"):
        summary_df = pd.DataFrame([raw]).T.rename(columns={0: "Value"})
        summary_df.index.name = "Feature"
        st.dataframe(
            summary_df.style.set_properties(**{
                'background-color': '#111827',
                'color': '#e2e8f0',
                'border-color': 'rgba(99,179,237,0.15)',
                'font-family': 'DM Sans',
            }),
            use_container_width=True
        )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-banner">
      <span style="font-size:1.2rem">⚕️</span>
      <span><strong>Medical Disclaimer:</strong> CKD Sentinel is intended for educational and research purposes only.
      Predictions should not be used as a substitute for professional medical diagnosis or clinical judgment.
      Always consult a qualified nephrologist or physician for medical advice.</span>
    </div>
    """, unsafe_allow_html=True)