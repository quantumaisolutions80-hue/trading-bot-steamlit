# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

# indicators
import ta

# data
import yfinance as yf

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# optional libs
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# MT5 (local only)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

# ----------------------------
# config & paths
# ----------------------------
st.set_page_config(page_title="Streamlit FX AI Bot", layout="wide")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MEMORY_FILE = DATA_DIR / "core_memory.json"
CSV_FILE = DATA_DIR / "gbpusd_m5.csv"
RAW_JSON = DATA_DIR / "gbpusd_m5_raw.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FUTURE_LOOKAHEAD = 6    # 6*5m = 30m lookahead for labelling
TP_POINTS = 0.0010
SL_POINTS = 0.0010

RF_ESTIMATORS = 100
LSTM_EPOCHS = 6
LSTM_BATCH = 64

AUTO_REFRESH_SECONDS = 1

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_bot")

# ----------------------------
# core memory (persistent)
# ----------------------------
def load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_memory(mem):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(mem, default=str, indent=2))

if "core_memory" not in st.session_state:
    st.session_state.core_memory = load_memory()
    st.session_state.core_memory.setdefault("last_download", None)
    st.session_state.core_memory.setdefault("rows", 0)
    st.session_state.core_memory.setdefault("models", {})
    st.session_state.core_memory.setdefault("metrics", {})
    st.session_state.core_memory.setdefault("open_trades", [])
    st.session_state.core_memory.setdefault("closed_trades", [])

# ----------------------------
# auto-refresh (robust cross-environment)
# ----------------------------
# meta tag for browser-based refresh:
st.markdown(f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">', unsafe_allow_html=True)

# fallback rerun guard to ensure consistent UI update
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
else:
    if time.time() - st.session_state.last_refresh >= AUTO_REFRESH_SECONDS:
        st.session_state.last_refresh = time.time()
        # NOTE: on some Streamlit versions rerun causes flicker; meta tag handles most cases
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ----------------------------
# utilities: MT5 (local)
# ----------------------------
def mt5_connect_local():
    if not MT5_AVAILABLE:
        return False, "MetaTrader5 package not installed in this environment."
    try:
        ok = mt5.initialize()
        if not ok:
            return False, f"MT5 initialize failed: {mt5.last_error()}"
        info = mt5.account_info()
        if info is None:
            return True, "MT5 initialized; terminal may not be logged in."
        return True, f"MT5 connected. Account: {info.login} | Balance: {info.balance}"
    except Exception as e:
        return False, f"MT5 connect error: {e}"

def mt5_get_rates(symbol="GBPUSD", timeframe=mt5.TIMEFRAME_M5, start=None, end=None):
    if not MT5_AVAILABLE:
        return None
    if start is None or end is None:
        return None
    utc_from = int(start.timestamp())
    utc_to = int(end.timestamp())
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'})[['Open','High','Low','Close','Volume']]
    return df

def mt5_place_order(symbol, lot=0.01, order_type="BUY", sl=None, tp=None, deviation=5):
    if not MT5_AVAILABLE:
        return False, "mt5 not available"
    if not mt5.initialize():
        return False, f"MT5 init error: {mt5.last_error()}"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False, f"{symbol} not in market watch"
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type=="BUY" else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if order_type=="BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 999999,
        "comment": "streamlit-bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    return result, getattr(result, "comment", str(result))

# ----------------------------
# data download (MT5 preferred, fallback to yfinance)
# ----------------------------
def download_gbpusd_m5(start_date_str="2020-01-01", end_date_str=None):
    if end_date_str is None:
        end_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)

    # try MT5 if available
    if MT5_AVAILABLE:
        ok, msg = mt5_connect_local()
        st.info(f"MT5: {msg}")
        try:
            df_mt5 = mt5_get_rates("GBPUSD", timeframe=mt5.TIMEFRAME_M5, start=start_dt, end=end_dt)
            if df_mt5 is not None and not df_mt5.empty:
                df_mt5.to_csv(CSV_FILE)
                df_mt5.reset_index().to_json(RAW_JSON, orient="records", date_format="iso")
                st.session_state.core_memory['last_download'] = datetime.utcnow().isoformat()
                st.session_state.core_memory['rows'] = len(df_mt5)
                save_memory(st.session_state.core_memory)
                return df_mt5
        except Exception as e:
            st.warning(f"MT5 download failed: {e}")

    # fallback to yfinance chunked 30-day downloads (5m)
    ticker = "GBPUSD=X"
    cur = start_dt
    frames = []
    while cur < end_dt:
        chunk_end = min(end_dt, cur + timedelta(days=30))
        try:
            df_chunk = yf.download(ticker, start=cur.strftime("%Y-%m-%d"), end=(chunk_end+timedelta(days=1)).strftime("%Y-%m-%d"), interval="5m", progress=False, threads=False)
            if df_chunk is not None and not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            st.warning(f"yfinance chunk failed {cur} -> {chunk_end}: {e}")
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.2)
    if frames:
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        # ensure standard column names
        df = df.rename(columns={c:c.capitalize() if c.islower() else c for c in df.columns})
        df.to_csv(CSV_FILE)
        df.reset_index().to_json(RAW_JSON, orient="records", date_format="iso")
        st.session_state.core_memory['last_download'] = datetime.utcnow().isoformat()
        st.session_state.core_memory['rows'] = len(df)
        save_memory(st.session_state.core_memory)
        return df
    else:
        st.error("No 5m data available from yfinance in that range. Try shorter range or use MT5 terminal.")
        return pd.DataFrame()

# ----------------------------
# indicators, features, labels
# ----------------------------
def compute_indicators(df):
    df = df.copy()
    # align column names
    df.columns = [c.capitalize() if c.islower() else c for c in df.columns]
    if not {"High","Low","Close"}.issubset(set(df.columns)):
        raise ValueError("OHLC columns missing")
    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["+di"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"], window=14)
    df["-di"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"], window=14)
    df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20)
    df["sma9"] = ta.trend.sma_indicator(df["Close"], window=9)
    df["ema30"] = ta.trend.ema_indicator(df["Close"], window=30)
    df["sma9_slope"] = df["sma9"].diff()
    df["ema30_slope"] = df["ema30"].diff()
    df["+di_slope"] = df["+di"].diff()
    return df

def create_labels(df, lookahead=FUTURE_LOOKAHEAD, tp=TP_POINTS, sl=SL_POINTS):
    df2 = df.reset_index()
    close = df2["Close"].values
    labels = np.zeros(len(close), dtype=int)
    for i in range(len(close)):
        end_i = min(len(close)-1, i+lookahead)
        win, lose = False, False
        for j in range(i+1, end_i+1):
            move = close[j] - close[i]
            if move >= tp:
                win = True
                break
            if move <= -sl:
                lose = True
                break
        labels[i] = 1 if win and not lose else 0
    df2["label"] = labels
    df2 = df2.set_index(df.index.name or df.index)
    return df2

def prepare_features(df):
    cols = ["adx","+di","-di","cci","sma9","ema30","sma9_slope","ema30_slope","+di_slope"]
    feat = df[cols].copy().fillna(method="ffill").fillna(0)
    feat["sma_diff"] = feat["sma9"] - feat["ema30"]
    return feat

# ----------------------------
# ML training
# ----------------------------
def train_models(df):
    df_ind = compute_indicators(df)
    df_lab = create_labels(df_ind)
    df_lab = df_lab.dropna()
    X = prepare_features(df_lab)
    y = df_lab["label"].astype(int)
    # time split
    split = int(len(X)*0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    metrics = {}
    # Random Forest
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    joblib.dump(rf, MODELS_DIR / "rf.joblib")
    pred_rf = rf.predict(X_te)
    acc_rf = accuracy_score(y_te, pred_rf)
    metrics["rf_acc"] = float(acc_rf)
    # XGBoost
    if XGB_AVAILABLE:
        xgbm = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
        xgbm.fit(X_tr, y_tr)
        xgbm.save_model(str(MODELS_DIR / "xgb.json"))
        pred_xgb = xgbm.predict(X_te)
        metrics["xgb_acc"] = float(accuracy_score(y_te, pred_xgb))
    # LSTM
    if TF_AVAILABLE:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_tr)
        Xs_test = scaler.transform(X_te)
        Xs = Xs.reshape((Xs.shape[0],1,Xs.shape[1]))
        Xs_test_seq = Xs_test.reshape((Xs_test.shape[0],1,Xs_test.shape[1]))
        model = Sequential()
        model.add(LSTM(64, input_shape=(Xs.shape[1], Xs.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(Xs, y_tr, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0)
        probs = model.predict(Xs_test_seq).reshape(-1)
        preds = (probs>0.5).astype(int)
        metrics["lstm_acc"] = float(accuracy_score(y_te, preds))
        model.save(str(MODELS_DIR / "lstm"))
        joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    st.session_state.core_memory["models"] = {"rf": str(MODELS_DIR / "rf.joblib")}
    st.session_state.core_memory["metrics"] = metrics
    save_memory(st.session_state.core_memory)
    return metrics

def load_models():
    models = {}
    rf_file = MODELS_DIR / "rf.joblib"
    if rf_file.exists():
        models["rf"] = joblib.load(rf_file)
    if XGB_AVAILABLE and (MODELS_DIR / "xgb.json").exists():
        booster = xgb.Booster()
        booster.load_model(str(MODELS_DIR / "xgb.json"))
        models["xgb"] = booster
    if TF_AVAILABLE and (MODELS_DIR / "lstm").exists():
        models["lstm"] = tf.keras.models.load_model(str(MODELS_DIR / "lstm"))
        models["lstm_scaler"] = joblib.load(MODELS_DIR / "lstm_scaler.pkl")
    return models

def predict_latest(models, df):
    df_ind = compute_indicators(df.tail(200))
    feat = prepare_features(df_ind).tail(1)
    X = feat.values
    out = {}
    if "rf" in models:
        out["rf"] = int(models["rf"].predict(X)[0])
    if "xgb" in models:
        out["xgb"] = int(models["xgb"].predict(xgb.DMatrix(X))[0])
    if "lstm" in models:
        scaler = models.get("lstm_scaler")
        Xs = scaler.transform(X) if scaler is not None else X
        Xs_seq = Xs.reshape((Xs.shape[0],1,Xs.shape[1]))
        proba = float(models["lstm"].predict(Xs_seq)[0][0])
        out["lstm_proba"] = proba
        out["lstm"] = int(proba>0.5)
    return out

# ----------------------------
# Backtester (simple)
# ----------------------------
def backtest_with_model(df, model_name="rf", verbose=False):
    """
    Simulate entries where model predicts 1 at bar t -> open market at t+1 close price,
    close when TP/SL hit within lookahead or at next opposite signal or end.
    This is a simple rule-based simulator not a high fidelity broker simulator.
    """
    if not CSV_FILE.exists():
        return {"error":"No data to backtest"}
    models = load_models()
    if model_name not in models:
        return {"error":f"Model {model_name} not found"}
    df_ind = compute_indicators(df)
    df_lab = create_labels(df_ind)
    feats = prepare_features(df_lab).fillna(0)
    n = len(feats)
    cash = 0.0
    trades = []
    i = 0
    while i < n-1:
        X = feats.iloc[i].values.reshape(1,-1)
        pred = None
        if model_name=="rf":
            pred = int(models["rf"].predict(X)[0])
        elif model_name=="xgb" and "xgb" in models:
            pred = int(models["xgb"].predict(xgb.DMatrix(X))[0])
        elif model_name=="lstm" and "lstm" in models:
            scaler = models.get("lstm_scaler")
            Xs = scaler.transform(X)
            Xs_seq = Xs.reshape((Xs.shape[0],1,Xs.shape[1]))
            proba = float(models["lstm"].predict(Xs_seq)[0][0])
            pred = int(proba>0.5)
        else:
            pred = 0
        if pred == 1:
            # open buy at next bar open/close; we'll use close price next bar as entry
            entry_idx = min(n-1, i+1)
            entry_price = df_lab["Close"].iloc[entry_idx]
            # look ahead to detect TP or SL
            closed = False
            for j in range(entry_idx+1, min(entry_idx+1+FUTURE_LOOKAHEAD, n)):
                p = df_lab["Close"].iloc[j]
                if p - entry_price >= TP_POINTS:
                    profit = p - entry_price
                    trades.append({"open_idx":entry_idx, "close_idx":j, "pl":profit})
                    cash += profit
                    closed = True
                    i = j
                    break
                if p - entry_price <= -SL_POINTS:
                    profit = p - entry_price
                    trades.append({"open_idx":entry_idx, "close_idx":j, "pl":profit})
                    cash += profit
                    closed = True
                    i = j
                    break
            if not closed:
                # close at end of lookahead
                end_j = min(entry_idx+FUTURE_LOOKAHEAD, n-1)
                p = df_lab["Close"].iloc[end_j]
                profit = p - entry_price
                trades.append({"open_idx":entry_idx, "close_idx":end_j, "pl":profit})
                cash += profit
                i = end_j
        i += 1
    # metrics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pl"]>0)
    losses = total_trades - wins
    net = cash
    win_rate = wins/total_trades if total_trades>0 else 0
    return {"trades":total_trades, "wins":wins, "losses":losses, "net":float(net), "win_rate":float(win_rate), "trades_list":trades}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("FX AI Trading Bot — GBPUSD M5")
col1, col2 = st.columns([2,1])

with col2:
    st.header("Controls")
    start = st.date_input("Start date", value=datetime(2020,1,1))
    end = st.date_input("End date", value=datetime.utcnow().date())
    download_btn = st.button("Download GBPUSD M5 (MT5 preferred)")
    train_btn = st.button("Train models (RF/XGB/LSTM)")
    backtest_btn = st.button("Backtest (RF)")
    predict_btn = st.button("Predict latest")
    st.markdown("---")
    st.write("MT5 Live (local only)")
    st.write(f"MT5 package available: {MT5_AVAILABLE}")
    check_mt5 = st.button("Check MT5 connection")
    lot = st.number_input("Lot size (local only)", min_value=0.01, value=0.01, step=0.01)
    buy_btn = st.button("Place BUY (local)")
    sell_btn = st.button("Place SELL (local)")
    st.markdown("---")
    st.write("Core memory")
    st.json(st.session_state.core_memory)

with col1:
    st.header("Data & Chart")
    if download_btn:
        with st.spinner("Downloading..."):
            df = download_gbpusd_m5(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if df is None or df.empty:
                st.error("Download failed or empty.")
            else:
                st.success(f"Downloaded {len(df)} rows.")
    else:
        if CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
            st.write(f"Loaded {len(df)} rows from local CSV")
        else:
            df = pd.DataFrame()
            st.info("No local data. Click Download to fetch data.")

    if not df.empty:
        st.line_chart(df["Close"].tail(500))

    st.header("Models & Predictions")
    if train_btn:
        if df.empty:
            st.warning("No data to train on. Download first.")
        else:
            with st.spinner("Training..."):
                metrics = train_models(df)
                st.success("Training finished")
                st.json(metrics)

    if predict_btn:
        if not CSV_FILE.exists():
            st.warning("No data file. Download first.")
        else:
            df_local = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
            models = load_models()
            if not models:
                st.warning("No models found. Train first.")
            else:
                preds = predict_latest(models, df_local)
                st.json(preds)

    if backtest_btn:
        if not CSV_FILE.exists():
            st.warning("No data file. Download first.")
        else:
            df_local = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
            res = backtest_with_model(df_local, model_name="rf")
            if "error" in res:
                st.error(res["error"])
            else:
                st.metric("Total trades", res["trades"])
                st.metric("Net P/L", round(res["net"],6))
                st.metric("Win rate", f"{res['win_rate']*100:.2f}%")
                st.write("Sample trades (first 20):")
                st.write(res["trades_list"][:20])

# MT5 buttons handling
if check_mt5:
    ok, msg = mt5_connect_local()
    st.write(msg)

if buy_btn or sell_btn:
    if not MT5_AVAILABLE:
        st.error("MT5 not available. Local MT5 & package required.")
    else:
        if not mt5.initialize():
            st.error(f"MT5 init error: {mt5.last_error()}")
        else:
            side = "BUY" if buy_btn else "SELL"
            res, info = mt5_place_order("GBPUSD", lot=lot, order_type=side)
            st.write("Order result:", info)
            st.session_state.core_memory.setdefault("open_trades", [])
            st.session_state.core_memory["open_trades"].append({"time":datetime.utcnow().isoformat(),"type":side,"lot":float(lot),"result":str(info)})
            save_memory(st.session_state.core_memory)

# final: show memory summary and models list
st.markdown("---")
st.subheader("Summary")
st.write("Last download:", st.session_state.core_memory.get("last_download"))
st.write("Rows:", st.session_state.core_memory.get("rows"))
st.write("Models saved:", os.listdir(MODELS_DIR) if MODELS_DIR.exists() else [])
st.write("Metrics:", st.session_state.core_memory.get("metrics"))
st.write("Open trades (recent):", st.session_state.core_memory.get("open_trades")[-5:])
st.write("Closed trades (recent):", st.session_state.core_memory.get("closed_trades")[-5:])
