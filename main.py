# File: main.py
from data.fetch_fundamental import get_fundamental_data_rti
from utils.indicators import add_technical_indicators
from utils.preprocess import preprocess_data
from model.train_model import train_lstm_model
import yfinance as yf

if __name__ == "__main__":
    ticker = "BBCA.JK"
    df = yf.download(ticker, period="12mo")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = add_technical_indicators(df)
    fund = get_fundamental_data_rti(ticker)
    X, y = preprocess_data(df, fund)
    train_lstm_model(X, y)
    print("Model dilatih dan disimpan.")