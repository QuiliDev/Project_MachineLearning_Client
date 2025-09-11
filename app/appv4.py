# app/app.py
from pathlib import Path
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from loaders import load_model_cfg, load_catalogs, load_features_sample
from utils import build_feature_row, suggest_actions
from explain import explain_one


st.set_page_config(
    page_title="Predicción de Disputas",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo (si existe)
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
if logo_path.exists():
    st.logo(str(logo_path), size="large")

# Recursos base
model, cfg = load_model_cfg()
cats = load_catalogs()
sample = load_features_sample()

thr_best_f1 = float(cfg.get("thresholds", {}).get("best_f1", 0.5)) if cfg else 0.5
thr_recall = cfg.get("thresholds", {}).get("recall_60", None)
thr_recall = float(thr_recall) if thr_recall is not None else None


def ensure_in_options(options, current):
    opts = list(options) if options is not None else []
    cur = "" if current is None else str(current)
    if cur and cur not in opts:
        opts = [cur] + opts
    return opts


def risk_band(p):
    if p < 0.40:
        return "bajo"
    if p < 0.70:
        return "medio"
    return "alto"


def style_risk(df):
    def _row_style(row):
        p = row.get("pred_proba_dispute", 0.0)
        if p >= 0.70:
            return ["background-color: #ffe5e5"] * len(row)
        if p >= 0.40:
            return ["background-color: #fff6db"] * len(row)
        return [""] * len(row)
    return df.style.apply(_row_style, axis=1).format({"pred_proba_dispute": "{:.1%}"})


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

page = st.sidebar.radio("Navegación", ["Inicio", "Predicción individual", "Scoring masivo"], index=0)


# ============================
# Página: INICIO (KPIs)
# ============================
if page == "Inicio":
    st.title("🚀 Predicción de Disputas – Panel")
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


# ============================
# Página: PREDICCIÓN INDIVIDUAL
# ============================
elif page == "Predicción individual":
    st.title("🧾 Predicción individual")

    def _init_state_from_row(row):
        st.session_state.Product = str(row["Product"])
        st.session_state.Sub_product = str(row["Sub_product"])
        st.session_state.Issue = str(row["Issue"])
        st.session_state.Sub_issue = str(row["Sub_issue"])
        st.session_state.Company = str(row["Company"])
        st.session_state.State = str(row["State"])
        st.session_state.ZIP_code = str(row["ZIP_code"])
        st.session_state.Date_received = pd.to_datetime(row["Date_received"]).date()
        st.session_state.Date_sent_to_company = pd.to_datetime(row["Date_sent_to_company"]).date()
        st.session_state.Company_response = str(row["Company_response"])

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
            st.session_state.Company_response = "Closed with explanation"

    top_actions = st.container()
    with st.container():
        cols = st.columns([1, 1, 1])
        if not sample.empty and cols[0].button("🎲 Ejemplo aleatorio", use_container_width=True):
            rnd = sample.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
            _init_state_from_row(rnd)
            st.toast("Ejemplo aleatorio cargado", icon="🎯")

    with st.form("form_prediccion", clear_on_submit=False):
        st.subheader("📌 Datos de la queja")
        col1, col2 = st.columns(2)

        with col1:
            prod_opts = ensure_in_options(cats.get("Product", []), st.session_state.get("Product"))
            st.selectbox("Product", options=prod_opts, key="Product")

            st.text_input("Sub-product", key="Sub_product")
            st.text_input("Issue", key="Issue")
            st.text_input("Sub-issue", key="Sub_issue")

            comp_opts = ensure_in_options(cats.get("Company", []), st.session_state.get("Company"))
            st.selectbox("Company", options=comp_opts, key="Company")

        with col2:
            state_opts = ensure_in_options(cats.get("State", []), st.session_state.get("State"))
            st.selectbox("State", options=state_opts, key="State")

            st.text_input("ZIP code", key="ZIP_code")
            st.date_input("Date received", key="Date_received")
            st.date_input("Date sent to company", key="Date_sent_to_company")

            resp_opts = ensure_in_options(
                cats.get("Company_response", [
                    "In progress", "Closed with explanation", "Closed with monetary relief",
                    "Closed with non-monetary relief", "Untimely response", "Other"
                ]),
                st.session_state.get("Company_response")
            )
            st.selectbox("Company response", options=resp_opts, key="Company_response")

        submitted = st.form_submit_button("Predecir riesgo", use_container_width=True)

    if submitted:
        row = build_feature_row(
            st.session_state.Product,
            st.session_state.Sub_product,
            st.session_state.Issue,
            st.session_state.Sub_issue,
            st.session_state.State,
            st.session_state.ZIP_code,
            st.session_state.Date_received,
            st.session_state.Date_sent_to_company,
            st.session_state.Company,
            st.session_state.Company_response
        )

        proba = float(model.predict_proba(row)[:, 1][0])
        label = int(proba >= threshold)

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

        try:
            st.subheader("Explicación (Top-5 SHAP)")
            expl = explain_one(model, row, topn=5)
            st.bar_chart(expl.set_index("feature")["shap_value"])
            st.dataframe(expl, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar SHAP: {e}")

        tips = suggest_actions(proba, int(row["Days_to_response"].iloc[0]), st.session_state.Company_response)
        with top_actions:
            st.subheader("Sugerencias de acción")
            for t in tips:
                st.write("• " + t)


# ============================
# Página: SCORING MASIVO
# ============================
elif page == "Scoring masivo":
    st.title("📦 Scoring masivo")
    st.caption("Sube un CSV con columnas originales del cliente o el formato de trabajo.")

    plantilla = pd.DataFrame([{
        "Complaint ID": 1290580,
        "Product": "Debt collection",
        "Sub-product": "Medical",
        "Issue": "Cont'd attempts collect debt not owed",
        "Sub-issue": "Debt is not mine",
        "State": "TX",
        "ZIP code": "77479",
        "Date received": "2015-03-19",
        "Date sent to company": "2015-03-19",
        "Company": "Accounts Receivable Consultants Inc.",
        "Company response": "Closed with explanation",
        "Timely response?": "Yes",
        "Consumer disputed?": "No"
    }])
    st.download_button(
        "📥 Descargar plantilla (formato original del cliente)",
        data=plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_cliente.csv",
        mime="text/csv",
        use_container_width=True
    )

    up = st.file_uploader("Sube tu CSV", type=["csv"])
    if up is not None:
        up.seek(0)
        try:
            raw = pd.read_csv(up, dtype=str, keep_default_na=False)
        except Exception:
            up.seek(0)
            raw = pd.read_csv(up, dtype=str, keep_default_na=False, engine="python")

        cols_trabajo = {
            "Product","Sub_product","Issue","Sub_issue","State","ZIP_code",
            "Date_received","Date_sent_to_company","Company","Company_response"
        }
        cols_original = {
            "Complaint ID","Product","Sub-product","Issue","Sub-issue","State",
            "ZIP code","Date received","Date sent to company","Company","Company response",
            "Timely response?","Consumer disputed?"
        }

        def normalize_original_df(df):
            df = df.rename(columns={
                "Sub-product": "Sub_product",
                "Sub-issue": "Sub_issue",
                "ZIP code": "ZIP_code",
                "Date received": "Date_received",
                "Date sent to company": "Date_sent_to_company",
                "Company response": "Company_response",
            })
            keep = ["Product","Sub_product","Issue","Sub_issue","State","ZIP_code",
                    "Date_received","Date_sent_to_company","Company","Company_response"]
            for c in keep:
                if c not in df.columns: df[c] = np.nan
            df["Date_received"] = pd.to_datetime(df["Date_received"], errors="coerce").fillna(pd.Timestamp("2015-01-01"))
            df["Date_sent_to_company"] = pd.to_datetime(df["Date_sent_to_company"], errors="coerce").fillna(pd.Timestamp("2015-01-01"))
            for c in ["Product","Sub_product","Issue","Sub_issue","State","ZIP_code","Company","Company_response"]:
                df[c] = df[c].astype(str).fillna("")
            return df[keep]

        def to_feature_block(df_norm):
            rows = []
            for _, r in df_norm.iterrows():
                rows.append(build_feature_row(
                    r["Product"], r["Sub_product"], r["Issue"], r["Sub_issue"],
                    r["State"], r["ZIP_code"], r["Date_received"].date(),
                    r["Date_sent_to_company"].date(), r["Company"], r["Company_response"]
                ))
            return pd.concat(rows, ignore_index=True)

        if cols_trabajo.issubset(set(raw.columns)):
            df_norm = raw.copy()
            if not np.issubdtype(df_norm["Date_received"].dtype, np.datetime64):
                df_norm["Date_received"] = pd.to_datetime(df_norm["Date_received"], errors="coerce")
            if not np.issubdtype(df_norm["Date_sent_to_company"].dtype, np.datetime64):
                df_norm["Date_sent_to_company"] = pd.to_datetime(df_norm["Date_sent_to_company"], errors="coerce")
            features = df_norm
        elif cols_original.issubset(set(raw.columns)):
            df_norm = normalize_original_df(raw)
            features = to_feature_block(df_norm)
        else:
            st.error("Las columnas no coinciden con el formato original ni con el de trabajo.")
            st.stop()

        probs = model.predict_proba(features)[:, 1]
        out = raw.copy()
        out["pred_proba_dispute"] = probs
        out["pred_label"] = (probs >= threshold).astype(int)
        out["risk_band"] = pd.cut(probs, bins=[-1, 0.4, 0.7, 1.01], labels=["bajo","medio","alto"])

        st.success(f"Filas procesadas: {len(out)}")
        st.dataframe(style_risk(out.head(30)), use_container_width=True)

        buf_csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📤 Descargar resultados (CSV)",
            data=buf_csv,
            file_name="predicciones_disputa.csv",
            mime="text/csv",
            use_container_width=True
        )

        bio = BytesIO()
        out.to_excel(bio, index=False)
        st.download_button(
            "📥 Descargar resultados (Excel)",
            data=bio.getvalue(),
            file_name="predicciones_disputa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.subheader("Distribución de riesgo")
        fig2, ax2 = plt.subplots()
        ax2.hist(probs, bins=20)
        ax2.set_xlabel("Probabilidad de disputa")
        ax2.set_ylabel("Frecuencia")
        st.pyplot(fig2, use_container_width=True)


st.divider()
st.caption("⚖️ Proyecto Académico – Predicción de Disputas • Anthony Quiliche")
