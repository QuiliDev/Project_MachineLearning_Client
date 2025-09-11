from io import BytesIO
import time
import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # añade la RAÍZ del repo

from app.loaders import load_model_cfg
from app.utils import build_feature_row

st.set_page_config(page_title="Scoring masivo", page_icon="📦", layout="centered")

model, cfg = load_model_cfg()

thr_best_f1 = float(cfg.get("thresholds", {}).get("best_f1", 0.5)) if cfg else 0.5
thr_recall = cfg.get("thresholds", {}).get("recall_60", None)
thr_recall = float(thr_recall) if thr_recall is not None else None

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
    "📥 Descargar plantilla (formato cliente)",
    data=plantilla.to_csv(index=False).encode("utf-8"),
    file_name="plantilla_cliente.csv",
    mime="text/csv",
    use_container_width=True
)

up = st.file_uploader("Sube tu CSV", type=["csv"])

# ---------------------------
# Helpers de transformación
# ---------------------------
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
        if c not in df.columns:
            df[c] = np.nan
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

def style_risk(df):
    def _row_style(row):
        p = row.get("pred_proba_dispute", 0.0)
        if p >= 0.70:
            return ["background-color: #ffe5e5"] * len(row)
        if p >= 0.40:
            return ["background-color: #fff6db"] * len(row)
        return [""] * len(row)
    return df.style.apply(_row_style, axis=1).format({"pred_proba_dispute": "{:.1%}"})


# -----------------------------------------------------------------
# PRELOADER: procesamiento con loader centrado mientras se trabaja
# -----------------------------------------------------------------
if up is not None:
    placeholder = st.empty()

    with placeholder.container():
        st.markdown(
            """
            <div style="display:flex;justify-content:center;align-items:center;height:280px;">
                <div style="text-align:center;">
                    <div class="loader"></div>
                    <p style="margin-top:14px;font-weight:600;">Procesando archivo, por favor espere...</p>
                </div>
            </div>
            <style>
            .loader {
              border: 12px solid #f3f3f3;
              border-top: 12px solid #3498db;
              border-radius: 50%;
              width: 64px;
              height: 64px;
              animation: spin 0.9s linear infinite;
              margin:auto;
            }
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    up.seek(0)
    try:
        raw = pd.read_csv(up, dtype=str, keep_default_na=False)
    except Exception:
        up.seek(0)
        raw = pd.read_csv(up, dtype=str, keep_default_na=False, engine="python")

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
        placeholder.empty()
        st.error("Las columnas no coinciden con el formato original ni con el de trabajo.")
        st.stop()

    probs = model.predict_proba(features)[:, 1]
    out = raw.copy()
    out["pred_proba_dispute"] = probs
    out["pred_label"] = (probs >= threshold).astype(int)
    out["risk_band"] = pd.cut(probs, bins=[-1, 0.4, 0.7, 1.01], labels=["bajo","medio","alto"])

    placeholder.empty()
    st.toast("Predicción realizada", icon="✨")
    st.success(f"✅ Archivo procesado correctamente. Filas: {len(out)}")
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
    with st.expander("📊 KPIs y bandas de riesgo", expanded=False):
        st.subheader("Distribución de riesgo")
        import matplotlib.pyplot as plt
        fig2, ax2 = plt.subplots()
        ax2.hist(probs, bins=20)
        ax2.set_xlabel("Probabilidad de disputa")
        ax2.set_ylabel("Frecuencia")
        st.pyplot(fig2, use_container_width=True)

        # ===================== [BLOQUE 1: KPIs + Bandas] =====================
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Riesgo medio (dataset)", f"{probs.mean():.1%}")
        kpi_col2.metric("Casos alto riesgo (≥0.70)", int((probs >= 0.70).sum()))
        kpi_col3.metric("Casos sobre umbral", int((probs >= threshold).sum()))

        st.subheader("Casos por banda de riesgo")
        band_counts = out["risk_band"].value_counts().reindex(["bajo","medio","alto"]).fillna(0).astype(int)
        st.bar_chart(band_counts)


        # ===================== [BLOQUE 4: Top compañías por riesgo] =====================
        if "Company" in out.columns:
            st.subheader("Top compañías por riesgo medio")
            grp = out.groupby("Company")["pred_proba_dispute"].agg(["mean","count"]).reset_index()
            grp = grp[grp["count"] >= 20].sort_values("mean", ascending=False).head(10)
            if not grp.empty:
                fig_topc, ax_topc = plt.subplots()
                ax_topc.barh(grp["Company"][::-1], (grp["mean"][::-1]*100.0))
                ax_topc.set_xlabel("Riesgo medio (%)")
                ax_topc.set_ylabel("Company")
                st.pyplot(fig_topc, use_container_width=True)
            else:
                st.info("No hay compañías con suficiente volumen (≥20) para ranking.")


st.divider()
st.caption("Proyecto Académico de ML – Predicción de disputas • Anthony Quiliche")
