# File: utils/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(df, fundamental_dict):
    df = df.copy()

    # Tambahkan semua metrik fundamental sebagai kolom konstan
    for key, val in fundamental_dict.items():
        if val is not None:
            df[key] = val
        else:
            df[key] = 0.0  # atau np.nan lalu dropna di bawah

    df.dropna(inplace=True)

    # Target: 1 jika harga besok naik, 0 jika turun atau sama
    target = (df['Close'].shift(-1) > df['Close']).astype(int).iloc[:-1]
    features = df.drop(columns=['Close']).iloc[:-1]

    # Skala semua fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = [], []
    window = 10
    for i in range(len(features_scaled) - window):
        X.append(features_scaled[i:i+window])
        y.append(target.iloc[i+window])

    return np.array(X), np.array(y)
