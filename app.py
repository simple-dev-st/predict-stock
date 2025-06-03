# File: app.py
import streamlit as st
import yfinance as yf
from data.fetch_fundamental import get_fundamental_data_rti
from utils.indicators import add_technical_indicators
from utils.preprocess import preprocess_data
from tensorflow.keras.models import load_model
import numpy as np

st.title("📈 Prediksi Arah Saham Indonesia")

# Input user
kode_saham = st.text_input("Masukkan kode saham (misal: BBCA.JK)", "BBCA.JK")
periode = st.selectbox("Periode data historis", [3, 6, 12, 24])

if st.button("Prediksi"):
    with st.spinner("Mengambil data..."):
        df = yf.download(kode_saham, period=f"{periode}mo")
        if df.empty:
            st.error("Data tidak ditemukan.")
        else:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = add_technical_indicators(df)
            fundamental = get_fundamental_data_rti(kode_saham)
            st.subheader("📊 Data Fundamental")
            st.write(fundamental)

            X, y = preprocess_data(df, fundamental)
            if len(X) == 0:
                st.warning("Data tidak cukup untuk prediksi.")
            else:
                model = load_model("model/model.h5")
                pred = model.predict(np.array([X[-1]]))[0]
                label = "Naik" if np.argmax(pred) == 1 else "Turun"
                confidence = round(100 * np.max(pred), 2)

                st.subheader("📌 Prediksi Besok")
                st.markdown(f"**{label}** ({confidence}%)")

                st.line_chart(df['Close'])