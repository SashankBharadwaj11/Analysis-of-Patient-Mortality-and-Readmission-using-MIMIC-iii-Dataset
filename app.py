import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# ----------------------------
# Load Models
# ----------------------------
mort_model = joblib.load('mortality_model.pkl')
readmit_model = joblib.load('readmission_catboost_model.pkl')

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tokenizer, model

tokenizer, bert_model = load_bert()

# ----------------------------
# UI Layout
# ----------------------------
st.title("🩺 Patient Risk Predictor (Mortality + Readmission)")

st.markdown("""
Doctors can enter structured patient data and paste clinical notes. The app will process both using CatBoost + ClinicalBERT.
""")

# --- Structured Inputs ---
st.header("📋 Patient Information")

col1, col2 = st.columns(2)
with col1:
    los = st.number_input("Length of Stay (days)", min_value=0, max_value=100, value=5, step=1)
    num_diag = st.number_input("Number of Diagnoses", min_value=0, max_value=20, value=2, step=1)
    num_proc = st.number_input("Number of Procedures", min_value=0, max_value=20, value=1, step=1)

with col2:
    sepsis = st.checkbox("Has Sepsis")
    diabetes = st.checkbox("Has Diabetes")
    vent = st.checkbox("Ventilated")

insurance = st.selectbox("Insurance", ["PRIVATE", "PUBLIC", "OTHER"])
discharge = st.selectbox("Discharge Group", ["HOME", "FACILITY", "DEATH", "OTHER"])
adm_type = st.selectbox("Admission Type", ["EMERGENCY", "URGENT", "ELECTIVE"])
adm_loc = st.selectbox("Admission Location", ["EMERGENCY ROOM ADMIT", "TRANSFER", "CLINIC REFERRAL"])
ethnicity = st.selectbox("Ethnicity", ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"])

# --- Clinical Note Input ---
st.header("📝 Clinical Summary Note")
note = st.text_area("Paste clinical note here", height=200)

# ----------------------------
# BERT Embedding Generator
# ----------------------------
def generate_embedding(note_text):
    tokens = tokenizer(note_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Predict Risk"):
    if len(note.strip()) == 0:
        st.warning("Please enter a clinical note.")
    else:
        st.info("Generating embeddings and running prediction...")

        # Structured inputs
        input_dict = {
            'LOS_DAYS': los,
            'NUM_DIAGNOSES': num_diag,
            'NUM_PROCEDURES': num_proc,
            'HAS_SEPSIS': int(sepsis),
            'HAS_DIABETES': int(diabetes),
            'HAS_VENT': int(vent),
            'INSURANCE_PRIVATE': int(insurance == 'PRIVATE'),
            'INSURANCE_PUBLIC': int(insurance == 'PUBLIC'),
            'INSURANCE_OTHER': int(insurance == 'OTHER'),
            'DISCHARGE_GROUP_OTHEROTHER': int(discharge == 'OTHER'),
            'DISCHARGE_GROUP_HOME': int(discharge == 'HOME'),
            'DISCHARGE_GROUP_DEATH': int(discharge == 'DEATH'),
            'DISCHARGE_GROUP_FACILITY': int(discharge == 'FACILITY'),
            'DISCHARGE_GROUP_OTHER': int(discharge == 'OTHER'),
            'ADMISSION_TYPE': adm_type,
            'ADMISSION_LOCATION': adm_loc,
            'ETHNICITY': ethnicity,
        }

        df = pd.DataFrame([input_dict])

        # Generate and attach embedding (with integer column names)
        embedding = generate_embedding(note)
        embedding_df = pd.DataFrame(embedding.reshape(1, -1), columns=[str(i) for i in range(768)])
        df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

        # Predict
        mort_prob = mort_model.predict_proba(df)[0][1]
        readm_prob = readmit_model.predict_proba(df)[0][1]

        # ----------------------------
        # Risk Tier Interpretation
        # ----------------------------

        # Mortality Risk Tier
        if mort_prob < 0.25:
            mort_risk = "🟢 Low risk of in-hospital mortality"
        elif mort_prob < 0.60:
            mort_risk = "🟡 Moderate risk of in-hospital mortality"
        else:
            mort_risk = "🔴 High risk of in-hospital mortality"

        # Readmission Risk Tier
        if readm_prob < 0.30:
            readm_risk = "🟢 Low likelihood of 30-day readmission"
        elif readm_prob < 0.60:
            readm_risk = "🟡 Moderate likelihood of 30-day readmission"
        else:
            readm_risk = "🔴 High likelihood of 30-day readmission"

        # ----------------------------
        # Display Results
        # ----------------------------
        st.subheader("📊 Prediction Results")

        st.markdown(f"""
        **🟥 Mortality Risk**
        - **Probability:** {mort_prob:.2%}
        - **Interpretation:** {mort_risk}
        """)

        st.markdown(f"""
        **🔁 Readmission Risk**
        - **Probability:** {readm_prob:.2%}
        - **Interpretation:** {readm_risk}
        """)