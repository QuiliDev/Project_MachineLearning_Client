# scripts/build_catalogs.py
import pandas as pd, yaml, os
from sklearn.preprocessing import LabelEncoder

RAW = "data/quejas-clientes.csv"
out_yaml = "models/catalogs.yaml"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(RAW, low_memory=False)

cols = ["Product","State","Company","Company response"]
df = df[cols].copy()

encoders = {}
mappings = {}
catalogs = {}

for col in cols:
    le = LabelEncoder()
    s = df[col].fillna("Desconocido").astype(str)
    le.fit(s)

    classes = list(le.classes_)
    catalogs[col.replace(" ", "_")] = classes  # para mostrar en la UI

    name2code = {name:int(le.transform([name])[0]) for name in classes}
    code2name = {int(v):k for k,v in name2code.items()}
    mappings[col.replace(" ", "_")] = {"name_to_code": name2code, "code_to_name": code2name}

with open(out_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump({"catalogs": catalogs, "mappings": mappings}, f, allow_unicode=True)

print(f"Guardado {out_yaml}")
