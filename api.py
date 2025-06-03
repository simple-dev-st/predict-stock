# api.py
import os
import numpy as np
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from data.fetch_fundamental import get_fundamental_data_rti
from utils.indicators import add_technical_indicators
from utils.preprocess import preprocess_data

app = FastAPI(title="Stock Direction Prediction API")

MODEL_PATH = "model/rf_model.pkl"

class PredictRequest(BaseModel):
    ticker: str
    period_months: int = 12

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

def train_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf

def load_or_train_model(X, y):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_model(X, y)

@app.post("/predict", response_model=PredictResponse)
def predict_stock_direction(req: PredictRequest):
    df = yf.download(req.ticker, period=f"{req.period_months}mo")
    if df.empty:
        raise HTTPException(status_code=404, detail="Data saham tidak ditemukan.")

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = add_technical_indicators(df)
    fundamental = get_fundamental_data_rti(req.ticker)

    X, y = preprocess_data(df, fundamental)
    if len(X) == 0:
        raise HTTPException(status_code=400, detail="Data tidak cukup untuk prediksi.")

    X_flat = X.reshape((X.shape[0], -1))  # Flatten untuk RandomForest
    model = load_or_train_model(X_flat, y)

    latest = X_flat[-1].reshape(1, -1)
    proba = model.predict_proba(latest)[0]
    label = "Naik" if np.argmax(proba) == 1 else "Turun"
    confidence = round(100 * np.max(proba), 2)

    return PredictResponse(prediction=label, confidence=confidence)

# Jalankan dengan: uvicorn api:app --reload
