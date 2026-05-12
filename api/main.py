from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Retail Price Optimizer API")

model = mlflow.pyfunc.load_model("models:/retail_price_optimizer/Production")

class PredictionInput(BaseModel):
    Quantity: float
    Month: int
    DayOfWeek: int
    Country_enc_United_Kingdom: int = 0
    Country_enc_Germany: int = 0
    Country_enc_EIRE: int = 0
    Country_enc_France: int = 0
    Country_enc_Netherlands: int = 0
    Country_enc_Spain: int = 0
    Country_enc_Belgium: int = 0
    Country_enc_Switzerland: int = 0
    Country_enc_Portugal: int = 0
    Country_enc_Sweden: int = 0
    Country_enc_Other: int = 0

class PredictionOutput(BaseModel):
    predicted_price: float

@app.get("/")
def root():
    return {"message": "Retail Price Optimizer API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    data = input_data.dict()
    df = pd.DataFrame([data])
    
    # Force l'ordre exact des colonnes comme à l'entraînement
    expected_cols = [
        "Quantity", "Month", "DayOfWeek",
        "Country_enc_Belgium", "Country_enc_EIRE", "Country_enc_France",
        "Country_enc_Germany", "Country_enc_Netherlands", "Country_enc_Other",
        "Country_enc_Portugal", "Country_enc_Spain", "Country_enc_Sweden",
        "Country_enc_Switzerland", "Country_enc_United_Kingdom"
    ]
    df = df[expected_cols]
    
    prediction = model.predict(df)[0]
    return {"predicted_price": round(float(prediction), 2)}