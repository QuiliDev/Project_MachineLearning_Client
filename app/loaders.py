from pathlib import Path
import joblib
import yaml
import pandas as pd
import numpy as np
from pandas.errors import ParserError
import streamlit as st

EXPECTED_ORIGINAL = [
    "Complaint ID","Product","Sub-product","Issue","Sub-issue","State",
    "ZIP code","Date received","Date sent to company","Company",
    "Company response","Timely response?","Consumer disputed?"
]

def _to_py(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    return obj

def load_model_cfg():
    root = Path(__file__).resolve().parent.parent
    model_path = root / "models" / "final_model_xgboost_predict_disputa.pkl"
    model = joblib.load(model_path)
    cfg = {}
    for name in ["model_config_xgboost_predict_disputa.yaml", "model_config.yaml"]:
        p = root / "models" / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                try:
                    cfg = yaml.safe_load(f)
                except yaml.constructor.ConstructorError:
                    cfg = yaml.unsafe_load(f)
            cfg = _to_py(cfg)
            break
    return model, cfg

def load_features_sample(n: int = 1000) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "transformado" / "quejas_features.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df.sample(min(n, len(df)), random_state=42)
    return pd.DataFrame()

def load_catalogs():
    """
    Devuelve un dict con:
      - display: listas de nombres para los select
      - to_code: mapeo nombre -> código
      - to_name: mapeo código -> nombre
    Lee models/catalogs.yaml si existe (soporta formatos con 'catalogs'/'mappings'
    o listas simples por columna). Si no existe, hace fallback con el dataset de features.
    """
    root = Path(__file__).resolve().parent.parent
    yml = root / "models" / "catalogs.yaml"

    display = {}
    to_code = {}
    to_name = {}

    if yml.exists():
        with open(yml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "catalogs" in data or "mappings" in data:
            cats = data.get("catalogs", {})
            maps = data.get("mappings", {})
            for col in ["Product", "State", "Company", "Company_response"]:
                names = cats.get(col, [])
                if not names and isinstance(data.get(col), list):
                    names = data[col]
                names = [str(x) for x in names if str(x).strip() != ""]
                names_sorted = sorted(set(names))
                display[col] = names_sorted

                n2c = maps.get(col, {}).get("name_to_code", {})
                if not n2c:
                    n2c = {name: i for i, name in enumerate(names_sorted)}
                c2n = {int(v): str(k) for k, v in n2c.items()}

                to_code[col] = {str(k): int(v) for k, v in n2c.items()}
                to_name[col] = c2n
        else:
            for col in ["Product", "State", "Company", "Company_response"]:
                names = data.get(col, [])
                names = [str(x) for x in names if str(x).strip() != ""]
                names_sorted = sorted(set(names))
                display[col] = names_sorted
                to_code[col] = {name: i for i, name in enumerate(names_sorted)}
                to_name[col] = {i: name for name, i in to_code[col].items()}

        return {"display": display, "to_code": to_code, "to_name": to_name}

    sample = load_features_sample()
    for col in ["Product", "State", "Company", "Company_response"]:
        if col in sample.columns:
            vals = sample[col].dropna().astype(str).unique().tolist()
            names_sorted = sorted(set(vals))
            display[col] = names_sorted
            to_code[col] = {name: i for i, name in enumerate(names_sorted)}
            to_name[col] = {i: name for name, i in to_code[col].items()}

    return {"display": display, "to_code": to_code, "to_name": to_name}

def read_client_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    except ParserError:
        pass

    uploaded_file.seek(0)
    try:
        return pd.read_csv(
            uploaded_file, dtype=str, keep_default_na=False,
            engine="python", sep=",", quotechar='"', doublequote=True
        )
    except ParserError:
        pass

    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, sep=";")
    except ParserError:
        pass

    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, engine="python", header=0, names=None)

    if set(EXPECTED_ORIGINAL).issubset(df_raw.columns):
        return df_raw

    cols = list(df_raw.columns)
    has_leading_idx = (cols[0].lower().startswith("unnamed")) or (cols[0] == "")
    base = 1 if has_leading_idx else 0
    try:
        fixed = []
        for _, r in df_raw.iterrows():
            row = [None]*len(EXPECTED_ORIGINAL)
            row[0]  = r.iloc[base + 0]
            row[1]  = r.iloc[base + 1]
            row[2]  = r.iloc[base + 2]
            row[3]  = r.iloc[base + 3]
            row[4]  = r.iloc[base + 4]
            row[5]  = r.iloc[base + 5]
            row[6]  = r.iloc[base + 6]
            row[7]  = r.iloc[base + 7]
            row[8]  = r.iloc[base + 8]
            tail3_start = len(r) - 3
            company_parts = r.iloc[base + 9:tail3_start]
            row[9]  = ",".join([str(x) for x in company_parts if str(x) != ""])
            row[10] = r.iloc[tail3_start + 0]
            row[11] = r.iloc[tail3_start + 1]
            row[12] = r.iloc[tail3_start + 2]
            fixed.append(row)
        return pd.DataFrame(fixed, columns=EXPECTED_ORIGINAL)
    except Exception:
        st.error("No pude interpretar el CSV. Re-exporta como CSV con campos entre comillas o comparte un ejemplo.")
        st.stop()
