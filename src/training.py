import os, joblib
from sklearn.pipeline import Pipeline

def train_model(preprocessor, model, X_train, y_train):
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    return pipe

def save_model(pipe, name, path="../models/"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"trained_model_{name}.pkl")
    joblib.dump(pipe, filename)
    return filename
