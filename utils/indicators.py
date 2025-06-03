# File: utils/indicators.py
import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def compute_stochastic(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=d_window).mean()
    return df

def compute_adx(df, window=14):
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ])
    df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                         df['High'] - df['High'].shift(), 0)
    df['+DM'] = df['+DM'].clip(lower=0)
    df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                         df['Low'].shift() - df['Low'], 0)
    df['-DM'] = df['-DM'].clip(lower=0)

    tr_smooth = df['TR'].rolling(window=window).mean()
    plus_di = 100 * (df['+DM'].rolling(window=window).mean() / tr_smooth)
    minus_di = 100 * (df['-DM'].rolling(window=window).mean() / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()
    df['ADX'] = adx
    return df

def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['Close'], window=14)

    # Volume-based
    df['VMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['Volume'] / df['VMA_20']
    df['OBV'] = compute_obv(df)

    # Stochastic Oscillator
    df = compute_stochastic(df)

    # ADX (trend strength)
    df = compute_adx(df)

    return df.dropna()