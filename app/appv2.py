import os
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from datetime import date

st.set_page_config(page_title="Predicción de Disputas", layout="centered")

@st.cache_data
def load_sample_data():
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "data" / "transformado" / "quejas_features.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame()

def _to_py(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    return obj

@st.cache_resource
def load_model_and_config():
    app_dir = Path(__file__).resolve().parent
    root_dir = app_dir.parent
    model_path = root_dir / "models" / "final_model_xgboost_predict_disputa.pkl"
    model = joblib.load(model_path)
    cfg = {}
    for name in ["model_config_xgboost_predict_disputa.yaml", "model_config.yaml"]:
        p = root_dir / "models" / name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.unsafe_load(f)
                cfg = _to_py(cfg)
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

def build_feature_row(
    Product, Sub_product, Issue, Sub_issue, State, ZIP_code,
    Date_received, Date_sent_to_company, Company, Company_response
):
    dr = pd.to_datetime(Date_received)
    ds = pd.to_datetime(Date_sent_to_company)
    days = max(int((ds - dr).days), 0)
    issue_len = len(str(Issue))
    sub_issue_words = len(str(Sub_issue).split())
    company_grouped = "Otras"
    days_bin = compute_bins(days)
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
        "Company_grouped": company_grouped,
        "Days_to_response_bin": days_bin
    }])
    return row

def predict_one(model, df_row, threshold=0.5):
    proba = float(model.predict_proba(df_row)[:, 1][0])
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
    df = pd.DataFrame({"feature": feat_names, "shap_value": vals})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(topn).drop(columns=["abs"])
    return df

def suggest_actions(prob, days, company_resp):
    tips = []
    if prob >= 0.5: tips.append("Priorizar el caso: riesgo alto de disputa.")
    if days > 3: tips.append("Responder en <24h y escalar a agente sénior.")
    if str(company_resp) in {"1","4",1,4}: tips.append("Revisar plantilla de respuesta; estos estados elevan el riesgo.")
    if not tips: tips.append("Riesgo bajo: mantener seguimiento estándar.")
    return tips

model, cfg = load_model_and_config()
thr_best_f1 = float(cfg.get("thresholds", {}).get("best_f1", 0.5)) if cfg else 0.5
thr_recall = cfg.get("thresholds", {}).get("recall_60", None)
thr_recall = float(thr_recall) if thr_recall is not None else None

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Umbral:", ["Equilibrado (F1)", "Priorizar recall", "Personalizado"])
    if mode == "Equilibrado (F1)":
        threshold = thr_best_f1
    elif mode == "Priorizar recall":
        threshold = thr_recall if thr_recall is not None else max(0.3, thr_best_f1 - 0.1)
    else:
        threshold = st.slider("Umbral de decisión", 0.05, 0.95, thr_best_f1, 0.01)
    st.markdown(f"**Umbral activo:** `{threshold:.2f}`")

st.title("Predicción de Disputa de Quejas")
tab1, tab2 = st.tabs(["Predicción individual", "Scoring por lote"])

with tab1:
    st.subheader("Ingresar una queja")
    col1, col2 = st.columns(2)

    sample_df = load_sample_data()
    random_row = None
    if not sample_df.empty:
        if st.button("🎲 Usar valores aleatorios del dataset", use_container_width=True):
            random_row = sample_df.sample(1).iloc[0]

    with col1:
        Product = st.text_input(
            "Product (código o texto)",
            str(random_row["Product"]) if random_row is not None else "Debt collection"
        )
        Sub_product = st.text_input(
            "Sub-product",
            str(random_row["Sub_product"]) if random_row is not None else "Medical"
        )
        Issue = st.text_input(
            "Issue",
            str(random_row["Issue"]) if random_row is not None else "Cont'd attempts collect debt not owed"
        )
        Sub_issue = st.text_input(
            "Sub-issue",
            str(random_row["Sub_issue"]) if random_row is not None else "Debt is not mine"
        )
        Company = st.text_input(
            "Company (código o texto)",
            str(random_row["Company"]) if random_row is not None else "Accounts Receivable Consultants Inc."
        )

    with col2:
        State = st.text_input(
            "State (código o texto)",
            str(random_row["State"]) if random_row is not None else "TX"
        )
        ZIP_code = st.text_input(
            "ZIP code",
            str(random_row["ZIP_code"]) if random_row is not None else "77479"
        )
        Date_received = st.date_input(
            "Date received",
            pd.to_datetime(random_row["Date_received"]).date() if random_row is not None else date(2015, 3, 19)
        )
        Date_sent_to_company = st.date_input(
            "Date sent to company",
            pd.to_datetime(random_row["Date_sent_to_company"]).date() if random_row is not None else date(2015, 3, 19)
        )

        if not sample_df.empty and "Company_response" in sample_df.columns:
            options_resp = sorted(sample_df["Company_response"].dropna().unique().tolist(), key=lambda x: str(x))
        else:
            options_resp = [0,1,2,3,4,5]
        if random_row is not None:
            try:
                idx_resp = options_resp.index(random_row["Company_response"])
            except ValueError:
                idx_resp = 0
        else:
            idx_resp = 1 if 1 in options_resp else 0

        Company_response = st.selectbox(
            "Company response (categoría original del dataset)",
            options_resp,
            index=idx_resp
        )

    if st.button("Predecir riesgo de disputa", use_container_width=True):
        df_row = build_feature_row(
            Product, Sub_product, Issue, Sub_issue, State, ZIP_code,
            Date_received, Date_sent_to_company, Company, Company_response
        )
        proba, label = predict_one(model, df_row, threshold=threshold)
        days = int(df_row.loc[0, "Days_to_response"])
        st.metric("Probabilidad de disputa", f"{proba:.1%}")
        st.markdown(f"**Etiqueta:** {'🔴 Disputa' if label==1 else '🟢 No disputa'} (umbral {threshold:.2f})")
        expl_df = explain_one(model, df_row, topn=5)
        st.subheader("Explicación (Top 5 features)")
        st.dataframe(expl_df, use_container_width=True)
        st.subheader("Sugerencias")
        for t in suggest_actions(proba, days, Company_response):
            st.write("• " + t)

with tab2:
    st.subheader("Subir CSV para scoring en lote")
    st.caption("Columnas esperadas: Product, Sub_product, Issue, Sub_issue, State, ZIP_code, Date_received, Date_sent_to_company, Company, Company_response")

    template = pd.DataFrame([{
        "Product": "Debt collection",
        "Sub_product": "Medical",
        "Issue": "Cont'd attempts collect debt not owed",
        "Sub_issue": "Debt is not mine",
        "State": "TX",
        "ZIP_code": "77479",
        "Date_received": "2015-03-19",
        "Date_sent_to_company": "2015-03-19",
        "Company": "Accounts Receivable Consultants Inc.",
        "Company_response": options_resp[0] if isinstance(options_resp, list) else 1
    }])
    st.download_button(
        "📥 Descargar plantilla CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_quejas.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded = st.file_uploader("Subir tu archivo CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        rows = []
        for _, r in raw.iterrows():
            rows.append(build_feature_row(
                r.get("Product",""),
                r.get("Sub_product",""),
                r.get("Issue",""),
                r.get("Sub_issue",""),
                r.get("State",""),
                r.get("ZIP_code",""),
                r.get("Date_received", date.today()),
                r.get("Date_sent_to_company", date.today()),
                r.get("Company",""),
                r.get("Company_response", options_resp[0] if isinstance(options_resp, list) else 1)
            ))
        batch = pd.concat(rows, ignore_index=True)
        probs = model.predict_proba(batch)[:, 1]
        out = raw.copy()
        out["pred_proba_dispute"] = probs
        out["pred_label"] = (probs >= threshold).astype(int)
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button(
            "📤 Descargar resultados",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_disputa.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()
st.caption("Proyecto Académico de ML – Predicción de disputas • Anthony Quiliche")
