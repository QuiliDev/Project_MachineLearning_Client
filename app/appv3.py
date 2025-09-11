import os
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from datetime import date

st.set_page_config(page_title="Predicción de Disputa", layout="centered")

@st.cache_resource
def load_model_and_config():
    app_dir = Path(__file__).resolve().parent
    root_dir = app_dir.parent
    model_path = root_dir / "models" / "final_model_xgboost_predict_disputa.pkl"
    model = joblib.load(model_path)
    cfg = {}
    for cand in ["model_config_xgboost_predict_disputa.yaml", "model_config.yaml"]:
        p = root_dir / "models" / cand
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            break
    return model, cfg

def compute_bins(days):
    if days <= 1: return "0-1 días"
    if days <= 3: return "2-3 días"
    if days <= 5: return "4-5 días"
    if days <= 10: return "6-10 días"
    if days <= 30: return "11-30 días"
    if days <= 100: return "31-100 días"
    return ">100 días"

def build_row(Product, Sub_product, Issue, Sub_issue, State, ZIP_code, Date_received, Date_sent_to_company, Company, Company_response):
    dr = pd.to_datetime(Date_received)
    ds = pd.to_datetime(Date_sent_to_company)
    days = max(int((ds - dr).days), 0)
    issue_len = len(str(Issue))
    sub_issue_words = len(str(Sub_issue).split())
    row = pd.DataFrame([{
        "Product": Product,
        "Sub_product": Sub_product,
        "Issue": Issue,
        "Sub_issue": Sub_issue,
        "State": State,
        "ZIP_code": str(ZIP_code),
        "Date_received": dr,
        "Date_sent_to_company": ds,
        "Company": Company,
        "Company_response": Company_response,
        "Timely_response": 0,
        "Consumer_disputed": 0,
        "target_disputed": 0,
        "Days_to_response": days,
        "Issue_len": issue_len,
        "Sub_issue_words": sub_issue_words,
        "Company_grouped": "Otras",
        "Days_to_response_bin": compute_bins(days)
    }])
    return row

def predict_one(model, df_row, threshold):
    proba = model.predict_proba(df_row)[:, 1][0]
    label = int(proba >= threshold)
    return proba, label

def explain_one(model, df_row, topn=5):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_trans = pre.transform(df_row)
    feat_names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    vals = shap_values[0]
    df = pd.DataFrame({"feature": feat_names, "shap_value": vals, "abs": np.abs(vals)}).sort_values("abs", ascending=False).head(topn).drop(columns=["abs"])
    return df

def suggest_actions(proba, days, company_response):
    tips = []
    if proba >= 0.5: tips.append("Priorizar caso: riesgo alto de disputa.")
    if days > 1: tips.append("Reducir el tiempo de respuesta (objetivo < 24h).")
    if str(company_response).lower() in ["in progress", "untimely response", "4"]:
        tips.append("Revisar calidad de la respuesta; usar guion estandarizado.")
    if not tips: tips.append("Mantener tiempos y calidad de respuesta actuales.")
    return tips

def make_template():
    return pd.DataFrame([{
        "Product": "Debt collection",
        "Sub_product": "Medical",
        "Issue": "Cont'd attempts collect debt not owed",
        "Sub_issue": "Debt is not mine",
        "State": "TX",
        "ZIP_code": "77479",
        "Date_received": "2015-03-19",
        "Date_sent_to_company": "2015-03-19",
        "Company": "Accounts Receivable Consultants Inc.",
        "Company_response": "Closed with explanation"
    }])

model, cfg = load_model_and_config()
thr_best_f1 = float(cfg.get("thresholds", {}).get("best_f1", 0.5))
thr_recall = cfg.get("thresholds", {}).get("recall_60", None)
thr_recall = float(thr_recall) if thr_recall is not None else None

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Modo de decisión", ["Equilibrado (F1)", "Priorizar recall", "Personalizado"], index=0)
    if mode == "Equilibrado (F1)":
        threshold = thr_best_f1
    elif mode == "Priorizar recall":
        threshold = thr_recall if thr_recall is not None else max(0.3, thr_best_f1 - 0.1)
    else:
        threshold = st.slider("Umbral", 0.05, 0.95, thr_best_f1, 0.01)
    st.caption(f"Umbral activo: {threshold:.2f}")

st.title("Predicción de Disputa de Quejas")

tab1, tab2, tab3 = st.tabs(["Predicción individual", "Scoring por lote", "Acerca del modelo"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        Product = st.text_input("Product", "Debt collection")
        Sub_product = st.text_input("Sub-product", "Medical")
        Issue = st.text_input("Issue", "Cont'd attempts collect debt not owed")
        Sub_issue = st.text_input("Sub-issue", "Debt is not mine")
        Company = st.text_input("Company", "Accounts Receivable Consultants Inc.")
    with col2:
        State = st.text_input("State", "TX")
        ZIP_code = st.text_input("ZIP code", "77479")
        Date_received = st.date_input("Date received", value=date(2015, 3, 19))
        Date_sent_to_company = st.date_input("Date sent to company", value=date(2015, 3, 19))
        Company_response = st.selectbox(
            "Company response",
            ["In progress", "Closed with explanation", "Closed with monetary relief", "Closed with non-monetary relief", "Untimely response", "Other"],
            index=1
        )
    if st.button("Predecir riesgo de disputa", use_container_width=True):
        row = build_row(Product, Sub_product, Issue, Sub_issue, State, ZIP_code, Date_received, Date_sent_to_company, Company, Company_response)
        proba, label = predict_one(model, row, threshold)
        st.metric("Probabilidad de disputa", f"{proba:.1%}")
        st.markdown(f"**Etiqueta:** {'🔴 Disputa' if label==1 else '🟢 No disputa'} (umbral {threshold:.2f})")
        expl = explain_one(model, row, topn=5)
        st.subheader("Explicación local (Top-5 SHAP)")
        st.dataframe(expl, use_container_width=True)
        days = int(row["Days_to_response"].iloc[0])
        tips = suggest_actions(proba, days, Company_response)
        st.subheader("Sugerencias")
        for t in tips:
            st.write("• " + t)

with tab2:
    st.caption("Sube un CSV con las columnas del formulario. Calcularemos features derivadas y devolveremos probabilidades y etiquetas.")
    uploaded = st.file_uploader("Subir CSV", type=["csv"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Descargar plantilla CSV", use_container_width=True):
            tpl = make_template()
            st.download_button("Descargar plantilla", tpl.to_csv(index=False).encode("utf-8"), "plantilla_quejas.csv", "text/csv", use_container_width=True)
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        rows = []
        for _, r in raw.iterrows():
            rows.append(build_row(
                r.get("Product",""),
                r.get("Sub_product",""),
                r.get("Issue",""),
                r.get("Sub_issue",""),
                r.get("State",""),
                r.get("ZIP_code",""),
                r.get("Date_received", date.today()),
                r.get("Date_sent_to_company", date.today()),
                r.get("Company",""),
                r.get("Company_response","Closed with explanation")
            ))
        batch = pd.concat(rows, ignore_index=True)
        probs = model.predict_proba(batch)[:,1]
        out = raw.copy()
        out["pred_proba_dispute"] = probs
        out["pred_label"] = (probs >= threshold).astype(int)
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("Descargar resultados", out.to_csv(index=False).encode("utf-8"), "predicciones_disputa.csv", "text/csv", use_container_width=True)

with tab3:
    st.markdown("**Modelo:** XGBoost dentro de un pipeline con preprocesamiento (OneHot + escalado).")
    st.markdown("**Umbral:** configurable según objetivo (equilibrado/recall).")
    st.markdown("**Interpretabilidad:** SHAP para explicación local por caso.")
