import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # añade la RAÍZ del repo


from app.loaders import load_model_cfg, load_features_sample
# ---- Barra lateral
with st.sidebar:
    st.header("📣 Navegación")
    st.success("Navega con las páginas de la izquierda.")

st.set_page_config(page_title="Predicción de Disputas", page_icon="⚖️", layout="centered")

st.title("⚖️ Predicción de Disputas")
st.caption("Proyecto de Machine Learning para predecir el riesgo de disputa en quejas de clientes.")



model, cfg = load_model_cfg()
sample = load_features_sample()

thr_best_f1 = float(cfg.get("thresholds", {}).get("best_f1", 0.5)) if cfg else 0.5
thr_crit = float(cfg.get("thresholds", {}).get("critical", 0.70)) if cfg else 0.70
thr_mod_low = float(cfg.get("thresholds", {}).get("moderate_low", 0.40)) if cfg else 0.40
thr_mod_high = float(cfg.get("thresholds", {}).get("moderate_high", 0.70)) if cfg else 0.70



if sample.empty:
    st.info("No se encontró muestra de datos transformados. Ve a **Scoring masivo** o **Predicción individual** para probar el modelo.")
else:
    subs = sample.sample(min(2000, len(sample)), random_state=42)
    probs = model.predict_proba(subs)[:, 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("📂 Quejas analizadas", "120k+")
    c2.metric("🔍 Variables consideradas", "25+")
    c3.metric("🎯 Modelo actual", "XGBoost")
    c1.metric("Riesgo promedio (muestra)", f"{probs.mean():.1%}")
    c2.metric("Casos críticos (≥{:.0%})".format(thr_crit), int((probs >= thr_crit).sum()))
    c3.metric("Casos moderados ({:.0%}–{:.0%})".format(thr_mod_low, thr_mod_high), int(((probs >= thr_mod_low) & (probs < thr_mod_high)).sum()))



    st.markdown("---")
    st.subheader("Distribución de riesgo (muestra)")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(probs, bins=20)
    ax.set_xlabel("Probabilidad de disputa")
    ax.set_ylabel("Frecuencia")
    ax.axvline(thr_best_f1, linestyle="--")
    ax.axvline(thr_crit, linestyle="--")
    st.pyplot(fig, clear_figure=True)

# ---- Call to Action
st.markdown(
    """
    ---
    👉 Usa el menú lateral para:
    - 🧑‍💻 **Predicción individual**: Completa un formulario con los datos de la queja.  
    - 📂 **Scoring masivo**: Sube un archivo CSV y obtén predicciones en lote.  
    - ❓ **Ayuda/About**: Consulta definiciones y ejemplos de interpretación.  

    ---
    """
)
st.markdown(
    """
    ### Cómo usar la herramienta
    - **Predicción individual**: completa el formulario o carga un ejemplo aleatorio y obtén el riesgo con explicación SHAP.
    - **Scoring masivo**: sube un CSV en formato original del cliente o en formato de trabajo e descarga los resultados.
    - **Ayuda / About**: definición de campos, umbrales y ejemplos de interpretación.
    """
)
st.info("Tip: Puedes comenzar cargando un ejemplo aleatorio en la sección **Predicción individual**.")

st.divider()
st.caption("Proyecto Académico de ML – Predicción de disputas • Anthony Quiliche")
