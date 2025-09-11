from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

from app.loaders import load_model_cfg, load_catalogs, load_features_sample
from app.utils import build_feature_row, predict_one, suggest_actions
from app.explain import explain_one

st.set_page_config(page_title="Predicción individual", page_icon="🧾", layout="centered")

model, cfg = load_model_cfg()
cats = load_catalogs()
sample = load_features_sample()

thr_best_f1 = float(cfg.get("thresholds", {}).get("best_f1", 0.5)) if cfg else 0.5
thr_recall = cfg.get("thresholds", {}).get("recall_60", None)
thr_recall = float(thr_recall) if thr_recall is not None else None

def opts_from_catalog(name, placeholder="— Selecciona —"):
    return [placeholder] + cats.get("display", {}).get(name, [])

def is_placeholder(v):
    return (v is None) or (str(v).strip() == "") or str(v).startswith("— ")

def code_to_name(col, code_val):
    m = cats.get("to_name", {}).get(col, {})
    try:
        return m[int(code_val)]
    except Exception:
        return str(code_val)

def name_to_code(col, name_val):
    m = cats.get("to_code", {}).get(col, {})
    return m.get(str(name_val), name_val)

with st.sidebar:
    st.header("Ajustes de decisión")
    mode = st.radio("Umbral", ["Equilibrado (F1)", "Priorizar recall", "Personalizado"], index=0)
    if mode == "Equilibrado (F1)":
        threshold = thr_best_f1
    elif mode == "Priorizar recall":
        threshold = thr_recall if thr_recall is not None else max(0.3, thr_best_f1 - 0.1)
    else:
        threshold = st.slider("Umbral de decisión", 0.05, 0.95, thr_best_f1, 0.01)
    st.caption(f"Umbral activo: **{threshold:.2f}**")

def _init_state_from_row(row):
    st.session_state.Product = code_to_name("Product", row["Product"])
    st.session_state.Sub_product = str(row["Sub_product"])
    st.session_state.Issue = str(row["Issue"])
    st.session_state.Sub_issue = str(row["Sub_issue"])
    st.session_state.Company = code_to_name("Company", row["Company"])
    st.session_state.State = code_to_name("State", row["State"])
    st.session_state.ZIP_code = str(row["ZIP_code"])
    st.session_state.Date_received = pd.to_datetime(row["Date_received"]).date()
    st.session_state.Date_sent_to_company = pd.to_datetime(row["Date_sent_to_company"]).date()
    st.session_state.Company_response = code_to_name("Company_response", row["Company_response"])

st.title("🧾 Predicción individual")

if "Product" not in st.session_state:
    if not sample.empty:
        _init_state_from_row(sample.iloc[0])
    else:
        st.session_state.Product = ""
        st.session_state.Sub_product = ""
        st.session_state.Issue = ""
        st.session_state.Sub_issue = ""
        st.session_state.Company = ""
        st.session_state.State = ""
        st.session_state.ZIP_code = ""
        st.session_state.Date_received = date(2015, 3, 19)
        st.session_state.Date_sent_to_company = date(2015, 3, 19)
        st.session_state.Company_response = ""

cols = st.columns([1, 1, 1])
if not sample.empty and cols[0].button("🎲 Cargar ejemplo aleatorio", use_container_width=True):
    rnd = sample.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
    _init_state_from_row(rnd)
    st.toast("Ejemplo aleatorio cargado", icon="🎯")

with st.form("form_prediccion", clear_on_submit=False):
    st.subheader("📌 Datos de la queja")
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Product", options=opts_from_catalog("Product"), key="Product")
        st.text_input("Sub-product", key="Sub_product")
        st.text_input("Issue", key="Issue")
        st.text_input("Sub-issue", key="Sub_issue")
        st.selectbox("Company", options=opts_from_catalog("Company"), key="Company")

    with col2:
        st.selectbox("State", options=opts_from_catalog("State"), key="State")
        st.text_input("ZIP code", key="ZIP_code")
        st.date_input("Date received", key="Date_received")
        st.date_input("Date sent to company", key="Date_sent_to_company")
        st.selectbox("Company response", options=opts_from_catalog("Company_response"), key="Company_response")

    submitted = st.form_submit_button("Predecir riesgo", use_container_width=True)

if submitted:
    missing = []
    for k in ["Product","Company","State","Company_response"]:
        if is_placeholder(st.session_state.get(k)):
            missing.append(k)
    if missing:
        st.warning("Completa los siguientes campos: " + ", ".join(missing))
        st.stop()

    prod_code = name_to_code("Product", st.session_state.Product)
    state_code = name_to_code("State", st.session_state.State)
    comp_code  = name_to_code("Company", st.session_state.Company)
    resp_code  = name_to_code("Company_response", st.session_state.Company_response)

    row = build_feature_row(
        prod_code,
        st.session_state.Sub_product,
        st.session_state.Issue,
        st.session_state.Sub_issue,
        state_code,
        st.session_state.ZIP_code,
        st.session_state.Date_received,
        st.session_state.Date_sent_to_company,
        comp_code,
        resp_code
    )

    proba, _ = predict_one(model, row, threshold)
    label = int(proba >= threshold)

    
    st.toast("Predicción realizada", icon="✨")
    st.subheader("Resultado")
    st.metric("Probabilidad de disputa", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    if proba < 0.40:
        st.success("🟢 Riesgo bajo")
    elif proba < 0.70:
        st.warning("🟡 Riesgo medio")
    else:
        st.error("🔴 Riesgo alto")
    st.caption(f"Etiqueta: {'Disputa' if label==1 else 'No disputa'} (umbral {threshold:.2f})")

    with st.expander("📊 KPIs y bandas de riesgo", expanded=False):
            # ===================== 1) GAUGE / VELOCÍMETRO =====================
        try:
            import plotly.graph_objects as go

            band_color = (
                "#2ecc71" if proba < 0.40 else
                "#f1c40f" if proba < 0.70 else
                "#e74c3c"
            )

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": band_color},
                    "steps": [
                        {"range": [0, 40], "color": "#eafaf1"},
                        {"range": [40, 70], "color": "#fff7e0"},
                        {"range": [70, 100], "color": "#ffe6e6"},
                    ],
                    "threshold": {
                        "line": {"color": "#34495e", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100
                    },
                },
                title={"text": "Velocímetro de riesgo (umbral marcado)"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 1.0))
            ax.barh([0], [100], color="#eafaf1")
            ax.barh([0], [30], color="#2ecc71")
            ax.barh([0], [70-40], left=40, color="#f1c40f")
            ax.barh([0], [100-70], left=70, color="#e74c3c")
            ax.axvline(proba*100, color="#2980b9", lw=3)
            ax.axvline(threshold*100, color="#34495e", lw=2, ls="--")
            ax.set_xlim(0,100); ax.set_yticks([]); ax.set_title("Riesgo (fallback)")
            st.pyplot(fig, use_container_width=True)


        # ===================== 3) DISTRIBUCIÓN HISTÓRICA + TU CASO =====================
        # Usa la muestra cargada con load_features_sample(); si está vacía, nos saltamos el gráfico.
        if not sample.empty:
            # Para evitar latencias, usamos una muestra de hasta 1500 filas
            sub = sample.sample(min(1500, len(sample)), random_state=42).copy()
            try:
                probs_hist = model.predict_proba(sub)[:, 1]
                fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
                ax_hist.hist(probs_hist, bins=20)
                ax_hist.axvline(proba, color="#e74c3c", lw=3, label=f"Tu caso ({proba:.2f})")
                ax_hist.axvline(threshold, color="#34495e", lw=2, ls="--", label=f"Umbral ({threshold:.2f})")
                ax_hist.set_xlabel("Probabilidad de disputa")
                ax_hist.set_ylabel("Frecuencia")
                ax_hist.set_title("Distribución histórica de riesgo (muestra)")
                ax_hist.legend()
                st.pyplot(fig_hist, use_container_width=True)
            except Exception as e:
                st.info(f"No se pudo calcular distribución histórica: {e}")

        # ===================== 4) RANKING SHAP (BARRAS HORIZONTALES) =====================
        try:
            expl = explain_one(model, row, topn=8)  # ya lo usas arriba; aquí lo formateamos distinto
            fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
            expl_sorted = expl.sort_values("shap_value")
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in expl_sorted["shap_value"]]
            ax_shap.barh(expl_sorted["feature"], expl_sorted["shap_value"], color=colors)
            ax_shap.axvline(0, color="#7f8c8d", lw=1)
            ax_shap.set_title("Impacto SHAP (positivo = empuja a disputa)")
            st.pyplot(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar el ranking SHAP: {e}")



        try:
            st.subheader("Explicación (Top-5 SHAP)")
            expl = explain_one(model, row, topn=5)
            st.bar_chart(expl.set_index("feature")["shap_value"])
            st.dataframe(expl, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar SHAP: {e}")

    tips = suggest_actions(proba, int(row["Days_to_response"].iloc[0]), st.session_state.Company_response)
    st.subheader("Sugerencias de acción")
    for t in tips:
        st.write("• " + t)

st.divider()
st.caption("Proyecto Académico de ML – Predicción de disputas • Anthony Quiliche")


