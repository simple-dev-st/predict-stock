# File: data/fetch_fundamental.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

def clean_number(text):
    text = text.replace(',', '').replace('%', '')
    try:
        return float(text)
    except:
        return None

def get_fundamental_data_rti(ticker):
    """Scrapes fundamental data from RTI Business for a given stock."""
    # RTI uses kode saham tanpa ".JK"
    kode = ticker.replace(".JK", "").upper()
    url = f"https://www.rti.co.id/idx/stock-summary/{kode}"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    data = {}
    try:
        table = soup.find("table")
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                val = cols[1].get_text(strip=True)
                data[key] = clean_number(val)
    except:
        return {}

    # Pilih metrik yang relevan
    return {
        "EPS": data.get("EPS"),
        "PER": data.get("PER"),
        "PBV": data.get("PBV"),
        "ROE": data.get("ROE"),
        "DER": data.get("DER"),
        "NPM": data.get("Net Profit Margin"),
        "ROA": data.get("ROA"),
        "Current Ratio": data.get("Current Ratio"),
        "Quick Ratio": data.get("Quick Ratio"),
        "Operating Margin": data.get("Operating Margin"),
        "Dividend Yield": data.get("Dividend Yield"),
        "Price to Sales": data.get("Price to Sales"),
        "Price to Free Cash Flow": data.get("Price to Free Cash Flow"),
        "Interest Coverage Ratio": data.get("Interest Coverage Ratio"),
        "PEG Ratio": data.get("PEG Ratio"),
        "BVPS": data.get("Book Value per Share (BVPS)"),
        "Market Cap": data.get("Market Capitalization")
    }