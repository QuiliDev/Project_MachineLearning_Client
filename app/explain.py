import shap, pandas as pd

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
    return df.sort_values("abs", ascending=False).head(topn).drop(columns=["abs"])
