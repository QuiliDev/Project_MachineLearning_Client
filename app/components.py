import streamlit as st
import pandas as pd
from app.utils import suggest_actions

def render_kpis(model, sample, cfg):
    st.header("📊 KPIs iniciales")
    if sample.empty:
        st.info("No hay datos de muestra disponibles")
        return
    probs = model.predict_proba(sample.sample(300))[:,1]
    st.metric("Promedio riesgo disputa", f"{probs.mean():.1%}")
    st.metric("Casos críticos (>0.7)", (probs>0.7).sum())

def render_form_single(model, cfg, cats, sample):
    st.subheader("Ingresar una queja")
    random_row = None
    if not sample.empty and st.button("🎲 Aleatorio del dataset"):
        random_row = sample.sample(1).iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        Product = st.selectbox("Product", cats.get("Product", []), index=0)
        Issue = st.text_input("Issue", random_row["Issue"] if random_row is not None else "")
        Company = st.selectbox("Company", cats.get("Company", []), index=0)
    with col2:
        State = st.selectbox("State", cats.get("State", []), index=0)
        ZIP = st.text_input("ZIP", str(random_row["ZIP_code"]) if random_row is not None else "")
        Company_response = st.selectbox("Company response", cats.get("Company_response", []), index=0)

    if st.button("Predecir riesgo"):
        # aquí iría el build_row() que ya tienes
        st.success("Predicción realizada (mockup)")
