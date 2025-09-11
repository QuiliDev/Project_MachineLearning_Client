import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from loaders import load_model_cfg, load_features_sample

st.set_page_config(page_title="Inicio", page_icon="🏠", layout="centered")

model, cfg = load_model_cfg()
sample = load_features_sample()

st.title("🚀 Panel de inicio")
if sample.empty:
    st.info("Sube datos en Scoring masivo o usa Predicción individual para probar el modelo.")
else:
    subs = sample.sample(min(800, len(sample)), random_state=42)
    probs = model.predict_proba(subs)[:, 1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Riesgo promedio (muestra)", f"{probs.mean():.1%}")
    col2.metric("Casos críticos (>70%)", int((probs > 0.7).sum()))
    col3.metric("Casos moderados (40–70%)", int(((probs >= 0.4) & (probs <= 0.7)).sum()))
    st.subheader("Distribución de riesgo (muestra)")
    fig, ax = plt.subplots()
    ax.hist(probs, bins=20)
    ax.set_xlabel("Probabilidad de disputa")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig, use_container_width=True)
    st.caption("KPIs calculados sobre muestra del dataset transformado para demo.")
