import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_loading():
    df = pd.read_excel("data/sample_test.xlsx")
    assert df.shape[0] > 0, "Le dataset est vide"
    assert "Price" in df.columns, "Colonne Price manquante"
    assert "Quantity" in df.columns, "Colonne Quantity manquante"
    print("Test 1 passed : dataset chargé correctement")

def test_preprocessing():
    df = pd.read_excel("data/sample_test.xlsx")
    df = df.dropna(subset=["Customer ID", "Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df[df["Price"] < 100]
    df["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.month
    df["DayOfWeek"] = pd.to_datetime(df["InvoiceDate"]).dt.dayofweek
    top_countries = df["Country"].value_counts().head(10).index
    df["Country_enc"] = df["Country"].apply(lambda x: x if x in top_countries else "Other")
    df = pd.get_dummies(df, columns=["Country_enc"])
    df.columns = df.columns.str.replace(" ", "_")
    assert df.shape[0] > 0, "Le preprocessing a tout supprimé"
    assert any(c.startswith("Country_enc_") for c in df.columns), "Encodage pays raté"
    print("Test 2 passed : preprocessing correct")

def test_model_training():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    df = pd.read_excel("data/sample_test.xlsx")
    df = df.dropna(subset=["Customer ID", "Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df[df["Price"] < 100]
    df["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.month
    df["DayOfWeek"] = pd.to_datetime(df["InvoiceDate"]).dt.dayofweek
    top_countries = df["Country"].value_counts().head(10).index
    df["Country_enc"] = df["Country"].apply(lambda x: x if x in top_countries else "Other")
    df = pd.get_dummies(df, columns=["Country_enc"])
    df.columns = df.columns.str.replace(" ", "_")

    feature_cols = ["Quantity", "Month", "DayOfWeek"] + \
                   [c for c in df.columns if c.startswith("Country_enc_")]
    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test), "Nombre de prédictions incorrect"
    assert all(p > 0 for p in preds), "Des prix négatifs prédits"
    print("Test 3 passed : modèle entraîné correctement")

def test_api():
    from fastapi.testclient import TestClient
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"))
    
    import unittest.mock as mock
    with mock.patch("mlflow.pyfunc.load_model") as mock_model:
        mock_model.return_value.predict = lambda x: [2.5]
        from api.main import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        print("Test 4 passed : API répond correctement")