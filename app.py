import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="FX AI Trading Bot", layout="wide")
st.title("📈 AI Forex Trading Bot – GBPUSD M5")

# Live auto refresh every 1 second
st.autorefresh(interval=1000, key="refresh")

DATA_FILE = "data.json"

# Load or create data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
        df = pd.DataFrame(data)
else:
    df = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

st.subheader("Latest Market Data")
st.dataframe(df.tail(20))

if len(df) > 10:
    fig = px.line(df, x="time", y="close", title="GBPUSD M5 Price")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Engine (Demo Mode)")
if len(df) > 20:
    last_close = df["close"].iloc[-1]
    direction = np.random.choice(["UP", "DOWN"])
    confidence = np.random.uniform(50, 100)

    st.metric("Predicted Direction", direction)
    st.metric("Confidence (%)", round(confidence, 2))
else:
    st.info("Waiting for data...")

st.success("✅ Streamlit UI Loaded Successfully")
