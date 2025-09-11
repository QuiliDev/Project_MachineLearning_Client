# app/utils.py
import pandas as pd
import numpy as np

def predict_one(model, df_row, threshold=0.5):
    proba = float(model.predict_proba(df_row)[:, 1][0])
    label = int(proba >= threshold)
    return proba, label


def compute_bins(days: int) -> str:
    if days <= 1: return "0-1 días"
    if days <= 3: return "2-3 días"
    if days <= 5: return "4-5 días"
    if days <= 10: return "6-10 días"
    if days <= 30: return "11-30 días"
    if days <= 100: return "31-100 días"
    return ">100 días"

def build_feature_row(Product, Sub_product, Issue, Sub_issue, State, ZIP_code,
                      Date_received, Date_sent_to_company, Company, Company_response):
    dr = pd.to_datetime(Date_received)
    ds = pd.to_datetime(Date_sent_to_company)
    days = max(int((ds - dr).days), 0)
    return pd.DataFrame([{
        "Product": str(Product),
        "Sub_product": str(Sub_product),
        "Issue": str(Issue),
        "Sub_issue": str(Sub_issue),
        "State": str(State),
        "ZIP_code": str(ZIP_code),
        "Date_received": dr,
        "Date_sent_to_company": ds,
        "Company": str(Company),
        "Company_response": str(Company_response),
        "Timely_response": 0,
        "Consumer_disputed": 0,
        "target_disputed": 0,
        "Days_to_response": days,
        "Issue_len": len(str(Issue)),
        "Sub_issue_words": len(str(Sub_issue).split()),
        "Company_grouped": "Otras",
        "Days_to_response_bin": compute_bins(days)
    }])

def suggest_actions(prob, days, company_resp: str):
    tips = []
    if prob >= 0.5: tips.append("Priorizar el caso: riesgo alto de disputa.")
    if days > 3: tips.append("Responder en <24h y escalar a agente sénior.")
    if company_resp in ("In progress", "Untimely response"): tips.append("Revisar plantilla de respuesta: aumenta riesgo.")
    if not tips: tips.append("Riesgo bajo: seguimiento estándar.")
    return tips

def select_index(options, value):
    try:
        return options.index(str(value))
    except Exception:
        return 0
