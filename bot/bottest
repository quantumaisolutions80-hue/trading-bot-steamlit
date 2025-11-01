import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

st.set_page_config(page_title="Simple Trading Bot", layout="wide")

st.title("✅ Simple Trading Bot (No ta-lib required)")

# Fake wallet balance
balance = 1000

# Function to get prices (Bitcoin fetch)
def get_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        data = requests.get(url).json()
        price = float(data['price'])
        return price
    except:
        return None

# Simple trading signals (no ta-lib)
def generate_signal(prices):
    if len(prices) < 5:
        return "NO-SIGNAL"

    sma_short = np.mean(prices[-3:])  # last 3 prices
    sma_long = np.mean(prices[-5:])   # last 5 prices

    if sma_short > sma_long:
        return "BUY"
    elif sma_short < sma_long:
        return "SELL"
    else:
        return "HOLD"

# Price tracking
if 'prices' not in st.session_state:
    st.session_state.prices = []

# Auto-refresh button
autorefresh = st.checkbox("Auto Refresh")

if autorefresh:
    time.sleep(3)
    st.experimental_rerun()

price = get_price()

if price:
    st.session_state.prices.append(price)
    st.write(f"📈 Latest Price: **${price:,.2f}**")
else:
    st.error("Error fetching price!")

# Show last prices
st.line_chart(st.session_state.prices)

# Signal
if len(st.session_state.prices) >= 5:
    signal = generate_signal(st.session_state.prices)
    st.subheader(f"🔔 Trading Signal: **{signal}**")
else:
    st.info("Collecting more price data...")

st.success("Bot running without ta-lib ✅")
