import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kidney Disease Predictor",
    page_icon="🫘",
    layout="wide",
)

st.markdown("""
<style>
    .result-ckd    { background:#ffe0e0; color:#c0392b; border:2px solid #c0392b;
                     padding:20px; border-radius:12px; text-align:center;
                     font-size:1.4rem; font-weight:bold; margin-top:16px; }
    .result-notckd { background:#d4edda; color:#155724; border:2px solid #27ae60;
                     padding:20px; border-radius:12px; text-align:center;
                     font-size:1.4rem; font-weight:bold; margin-top:16px; }
    .section-title { font-size:1.05rem; font-weight:700; color:#2c3e50;
                     border-left:4px solid #3498db; padding-left:8px;
                     margin:18px 0 8px; }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_artifacts():
    # Build paths INSIDE the function, relative to this script's location
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
    st.error(
        f"❌ Could not load model files: {e}\n\n"
        "Make sure `model_gnb.pkl` and `encoding.pkl` exist in `outputs/models/`."
    )

# # ── Feature columns (id & classification dropped) ─────────────────────────────
FEATURE_COLS = ['age','bp','sg','al','su','rbc','pc','pcc','ba',
                'bgr','bu','sc','sod','pot','hemo','pcv','wc','rc',
                'htn','dm','cad','appet','pe','ane']

# CAT_COLS = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
# NUM_COLS = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot',
#             'hemo','pcv','wc','rc']

# # Encoding maps matching training data
# CAT_MAP = {
#     'rbc':   {'normal': 0, 'abnormal': 1},
#     'pc':    {'normal': 0, 'abnormal': 1},
#     'pcc':   {'notpresent': 0, 'present': 1},
#     'ba':    {'notpresent': 0, 'present': 1},
#     'htn':   {'no': 0, 'yes': 1},
#     'dm':    {'no': 0, 'yes': 1},
#     'cad':   {'no': 0, 'yes': 1},
#     'appet': {'poor': 0, 'good': 1},
#     'pe':    {'no': 0, 'yes': 1},
#     'ane':   {'no': 0, 'yes': 1},
# }

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🫘 Chronic Kidney Disease Predictor")
st.caption("Enter patient lab values below. Fields marked **optional** can be left blank if unknown.")

with st.form("predict_form"):

    st.markdown('<div class="section-title">👤 Demographics & Vitals</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", min_value=1, max_value=120, value=48)
    bp  = c2.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=180, value=80)

    st.markdown('<div class="section-title">🧪 Urine Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    sg  = c1.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
    al  = c2.selectbox("Albumin (0–5)", [0, 1, 2, 3, 4, 5], index=1)
    su  = c3.selectbox("Sugar (0–5)", [0, 1, 2, 3, 4, 5], index=0)

    c1, c2, c3, c4 = st.columns(4)
    rbc = c1.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc  = c2.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = c3.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    ba  = c4.selectbox("Bacteria", ["notpresent", "present"])

    st.markdown('<div class="section-title">🩸 Blood Tests</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    bgr  = c1.number_input("Blood Glucose Random (mgs/dl)", 50, 500, 121)
    bu   = c2.number_input("Blood Urea (mgs/dl)", 1, 400, 36)
    sc   = c3.number_input("Serum Creatinine (mgs/dl)", 0.1, 40.0, 1.2, step=0.1)
    hemo = c4.number_input("Haemoglobin (gms)", 1.0, 20.0, 15.4, step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    pcv  = c1.number_input("Packed Cell Volume", 10, 60, 44)
    wc   = c2.number_input("WBC Count (cells/cumm)", 2000, 26400, 7800)
    rc   = c3.number_input("RBC Count (millions/cmm)", 0.0, 8.0, 5.2, step=0.1)

    st.markdown('<div class="section-title">🧂 Electrolytes (optional)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    sod_input = c1.text_input("Sodium (mEq/L) — leave blank if unknown", value="")
    pot_input = c2.text_input("Potassium (mEq/L) — leave blank if unknown", value="")

    st.markdown('<div class="section-title">📋 Medical History</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    htn   = c1.selectbox("Hypertension", ["yes", "no"])
    dm    = c2.selectbox("Diabetes Mellitus", ["yes", "no"])
    cad   = c3.selectbox("Coronary Artery Disease", ["no", "yes"])

    c1, c2, c3 = st.columns(3)
    appet = c1.selectbox("Appetite", ["good", "poor"])
    pe    = c2.selectbox("Pedal Edema", ["no", "yes"])
    ane   = c3.selectbox("Anaemia", ["no", "yes"])

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# ── Predict ────────────────────────────────────────────────────────────────────
if submitted and model_loaded:
    sod = float(sod_input) if sod_input.strip() != "" else np.nan
    pot = float(pot_input) if pot_input.strip() != "" else np.nan

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

    # Transform features using the encoder
    df_transformed = label_encoder.transform(df_input)

    # Predict
    pred  = model.predict(df_transformed)[0]
    proba = model.predict_proba(df_transformed)[0] if hasattr(model, "predict_proba") else None

    # ── Decode label safely ──────────────────────────────────────────────
    # The model outputs 0/1 or 'ckd'/'notckd' — map directly
    LABEL_MAP = {0: "notckd", 1: "ckd", "ckd": "ckd", "notckd": "notckd"}
    label = LABEL_MAP.get(str(pred).lower(), str(pred))
    is_ckd = label == "ckd"

    # ── Class names for probabilities ────────────────────────────────────
    # Try to get classes from model directly (works for sklearn models)
    if hasattr(model, "classes_"):
        classes = [str(c).lower() for c in model.classes_]
    else:
        classes = ["notckd", "ckd"]  # fallback — adjust order if needed

    st.markdown("---")
    st.subheader("📊 Result")

    if is_ckd:
        st.markdown('<div class="result-ckd">⚠️ Chronic Kidney Disease (CKD) Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-notckd">✅ Not Chronic Kidney Disease (Not CKD)</div>', unsafe_allow_html=True)

    if proba is not None:
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability (%)": [round(p * 100, 2) for p in proba]
        })
        st.markdown("#### Prediction Probabilities")
        c1, c2 = st.columns(2)
        for i, row in prob_df.iterrows():
            (c1 if i == 0 else c2).metric(row["Class"].upper(), f"{row['Probability (%)']:.1f}%")

        ckd_prob = proba[classes.index("ckd")] * 100 if "ckd" in classes else 0
        st.progress(int(ckd_prob))

    with st.expander("📋 Input Summary"):
        st.dataframe(pd.DataFrame([raw]).T.rename(columns={0: "Value"}), use_container_width=True)

    if hasattr(model, "feature_importances_"):
        with st.expander("🔬 Top Contributing Features"):
            fi = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)
            st.bar_chart(fi.set_index("Feature"))

    st.info("⚕️ **Disclaimer:** This tool is for educational/research purposes only. Always consult a qualified medical professional.")