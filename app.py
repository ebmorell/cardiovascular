import os
import gdown
import cloudpickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🔗 DESCARGAR MODELO Y VARIABLES DESDE GOOGLE DRIVE
model_url = "https://drive.google.com/file/d/1W1YP0NbDCC78VGcKXwvxXWWqQGQ-13v1/view?usp=drive_link"
features_url = "https://drive.google.com/file/d/1J14Tgotjiszcmu5ovY4N4Rsh911nluyz/view?usp=drive_link"

if not os.path.exists("rsf_model.pkl"):
    gdown.download(model_url, "rsf_model.pkl", quiet=False)

if not os.path.exists("model_features.pkl"):
    gdown.download(features_url, "model_features.pkl", quiet=False)

with open("rsf_model.pkl", "rb") as f:
    rsf = cloudpickle.load(f)
model_features = joblib.load("model_features.pkl")

st.title("🧠 Riesgo Cardiovascular a 5 Años en Pacientes con VIH")
st.markdown("""Introduce las variables clínicas del paciente para estimar el riesgo de un evento cardiovascular en los próximos 5 años.""")

def prepare_input():
    input_dict = {}
    input_dict["Age"] = st.number_input("Edad", min_value=0, max_value=100, value=45)
    input_dict["CD4_Nadir"] = st.number_input("CD4 nadir", min_value=0, value=350)
    input_dict["CD8_Nadir"] = st.number_input("CD8 nadir", min_value=0, value=1000)
    input_dict["CD4_CD8_Ratio"] = st.number_input("CD4/CD8 Ratio", min_value=0.0, value=0.5)
    input_dict["Cholesterol"] = st.number_input("Colesterol total (mg/dL)", value=180.0)
    input_dict["HDL"] = st.number_input("HDL (mg/dL)", value=50.0)
    input_dict["Triglycerides"] = st.number_input("Triglicéridos (mg/dL)", value=150.0)
    input_dict["Non_HDL_Cholesterol"] = st.number_input("Colesterol no HDL (mg/dL)", value=130.0)
    input_dict["Triglyceride_HDL_Ratio"] = st.number_input("Relación TG/HDL", value=3.0)

    cat_vars = {
        "Sex": ["Man", "Woman"],
        "Transmission_mode": ["Homo/Bisexual", "Injecting Drug User", "Heterosexual", "Other or Unknown"],
        "Origin": ["Spain", "Not Spain"],
        "Education_Level": ["No studies", "Primary", "Secondary/High School", "University", "Other/Unknown"],
        "AIDS": ["No", "Yes"],
        "Viral_Load": ["< 100.000 copies/ml", "≥ 100.000 copies/ml"],
        "ART": ["2NRTI+1NNRTI", "2NRTI+1IP", "2NRTI+1II", "Other"],
        "Hepatitis_C": ["Negative", "Positive"],
        "Anticore_HBV": ["Negative", "Positive"],
        "HBP": ["No", "Yes"],
        "Smoking": ["No Smoking", "Current Smoking", "Past Smoking"],
        "Diabetes": ["No", "Yes"]
    }

    for var, options in cat_vars.items():
        selected = st.selectbox(var.replace("_", " "), options)
        for opt in options[1:]:
            col_name = f"{var}_{opt}"
            input_dict[col_name] = 1 if selected == opt else 0

    return pd.DataFrame([input_dict])

input_df = prepare_input()
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

if st.button("Calcular riesgo"):
    surv_fn = rsf.predict_survival_function(input_df)[0]
    risk_5y = 1 - surv_fn(5)
    st.subheader(f"🔎 Riesgo estimado de evento cardiovascular a 5 años: {risk_5y:.2%}")
    st.subheader("📈 Curva de Supervivencia Estimada")
    fig, ax = plt.subplots()
    ax.plot(surv_fn.x, surv_fn.y, label="Supervivencia")
    ax.axvline(x=5, color='red', linestyle='--', label="5 años")
    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel("Probabilidad de no tener evento")
    ax.legend()
    st.pyplot(fig)
