import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = [
    "Product","Sub_product","Issue","Sub_issue","State","Company",
    "Company_response","Company_grouped","Days_to_response_bin","ZIP_code"
]
NUMERIC_COLS = ["Days_to_response","Issue_len","Sub_issue_words"]

def load_dataset(path="../data/transformado/quejas_features.csv"):
    df = pd.read_csv(path, low_memory=False, parse_dates=["Date_received","Date_sent_to_company"])
    return df

def make_features_targets(df):
    X = df.drop(columns=["Consumer_disputed","target_disputed","Timely_response"])
    y_timely = df["Timely_response"].astype(int)
    y_disputed = df["target_disputed"].astype(int)
    return X, y_timely, y_disputed

def build_preprocessor(categorical_cols=CATEGORICAL_COLS, numeric_cols=NUMERIC_COLS):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre

def get_feature_lists():
    return CATEGORICAL_COLS, NUMERIC_COLS
