# app.py - Pattern Learning FX Bot (No TA dependency)
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

# data
import yfinance as yf

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
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
st.set_page_config(page_title="Pattern Learning FX Bot", layout="wide")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PATTERNS_DIR = Path("patterns")
MEMORY_FILE = DATA_DIR / "core_memory.json"
CSV_FILE = DATA_DIR / "gbpusd_m5.csv"
PATTERNS_FILE = PATTERNS_DIR / "winning_patterns.json"
BUY_PATTERNS_FILE = PATTERNS_DIR / "buy_patterns.json"
SELL_PATTERNS_FILE = PATTERNS_DIR / "sell_patterns.json"
HIGH_TREND_FILE = PATTERNS_DIR / "high_trend_patterns.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

FUTURE_LOOKAHEAD = 6    # 6*5m = 30m lookahead
TP_POINTS = 0.0010
SL_POINTS = 0.0010
MIN_PROFIT_TO_SAVE = 0.0005  # 5 pips minimum to save pattern
HIGH_TREND_ADX_THRESHOLD = 25

RF_ESTIMATORS = 100
LSTM_EPOCHS = 6
LSTM_BATCH = 64

AUTO_REFRESH_SECONDS = 60

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pattern_bot")

# ----------------------------
# Custom Indicators (no TA library)
# ----------------------------
def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def cci(high, low, close, period=20):
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci_val = (tp - sma_tp) / (0.015 * mad)
    return cci_val

def true_range(high, low, close):
    """True Range for ATR calculation"""
    h_l = high - low
    h_pc = np.abs(high - close.shift(1))
    l_pc = np.abs(low - close.shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    """Average True Range"""
    tr = true_range(high, low, close)
    return tr.rolling(window=period).mean()

def bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands"""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def directional_movement(high, low, close, period=14):
    """ADX, +DI, -DI calculation"""
    # Plus and Minus Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # When both are positive, only the larger one is kept
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    
    # True Range
    tr = true_range(high, low, close)
    
    # Smoothed TR and DM
    atr_val = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
    
    # ADX calculation
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()
    
    return adx_val, plus_di, minus_di

def compute_indicators(df):
    """Compute all indicators without TA library"""
    df = df.copy()
    df.columns = [col.strip().title() if isinstance(col, str) else col for col in df.columns]
    
    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns. Have: {df.columns.tolist()}, Need: {required_cols}")
    
    # Moving averages
    df["sma9"] = sma(df["Close"], 9)
    df["ema30"] = ema(df["Close"], 30)
    df["sma9_slope"] = df["sma9"].diff()
    df["ema30_slope"] = df["ema30"].diff()
    
    # CCI
    df["cci"] = cci(df["High"], df["Low"], df["Close"], period=20)
    
    # ADX with period 8 as requested
    df["adx"], df["+di"], df["-di"] = directional_movement(df["High"], df["Low"], df["Close"], period=8)
    df["+di_slope"] = df["+di"].diff()
    df["-di_slope"] = df["-di"].diff()
    
    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(df["Close"], period=20)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # ATR
    df["atr"] = atr(df["High"], df["Low"], df["Close"], period=14)
    
    return df

# ----------------------------
# Pattern storage and matching
# ----------------------------
def save_patterns(patterns, filename):
    """Save patterns to JSON file"""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(patterns, f, indent=2, default=str)

def load_patterns(filename):
    """Load patterns from JSON file"""
    if filename.exists():
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading patterns from {filename}: {e}")
            return []
    return []

def extract_pattern_features(df, idx):
    """Extract indicator values at specific index as pattern signature"""
    feature_cols = ["sma9", "ema30", "cci", "adx", "+di", "-di", 
                   "bb_position", "bb_width", "atr", "sma9_slope", "ema30_slope"]
    
    pattern = {}
    for col in feature_cols:
        if col in df.columns:
            pattern[col] = float(df[col].iloc[idx])
    
    # Additional context
    pattern["close"] = float(df["Close"].iloc[idx])
    pattern["timestamp"] = str(df.index[idx])
    
    return pattern

def calculate_pattern_similarity(pattern1, pattern2):
    """Calculate similarity score between two patterns (0-1, higher is more similar)"""
    feature_keys = ["sma9_slope", "ema30_slope", "cci", "adx", "+di", "-di", "bb_position", "atr"]
    
    differences = []
    for key in feature_keys:
        if key in pattern1 and key in pattern2:
            # Normalize by typical ranges
            if key == "cci":
                diff = abs(pattern1[key] - pattern2[key]) / 200  # CCI range ~-200 to 200
            elif key in ["adx", "+di", "-di"]:
                diff = abs(pattern1[key] - pattern2[key]) / 100  # 0-100 range
            elif key == "bb_position":
                diff = abs(pattern1[key] - pattern2[key])  # Already 0-1
            elif key == "atr":
                diff = abs(pattern1[key] - pattern2[key]) / 0.01  # Normalize by typical ATR
            else:
                diff = abs(pattern1[key] - pattern2[key]) / 0.001  # For slopes
            
            differences.append(min(diff, 1.0))
    
    if not differences:
        return 0
    
    # Average difference, then convert to similarity
    avg_diff = np.mean(differences)
    similarity = 1 - avg_diff
    return max(0, similarity)

def find_similar_patterns(current_pattern, pattern_library, top_k=5, min_similarity=0.7):
    """Find similar patterns from library"""
    similarities = []
    
    for stored_pattern in pattern_library:
        sim = calculate_pattern_similarity(current_pattern, stored_pattern)
        if sim >= min_similarity:
            similarities.append((stored_pattern, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# ----------------------------
# Core memory
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
    st.session_state.core_memory.setdefault("patterns_count", {"buy": 0, "sell": 0, "high_trend": 0})

# ----------------------------
# Auto-refresh
# ----------------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
if elapsed >= AUTO_REFRESH_SECONDS:
    st.session_state.last_refresh = time.time()
    time.sleep(0.1)
    st.rerun()

# ----------------------------
# Data download
# ----------------------------
def download_gbpusd_m5(start_date_str="2020-01-01", end_date_str=None):
    if end_date_str is None:
        end_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)

    # Try MT5 if available (keeping original MT5 code)
    if MT5_AVAILABLE:
        try:
            if mt5.initialize():
                symbol = "GBPUSD"
                timeframe = mt5.TIMEFRAME_M5
                utc_from = int(start_dt.timestamp())
                utc_to = int(end_dt.timestamp())
                rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['datetime'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('datetime', inplace=True)
                    df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'})
                    df = df[['Open','High','Low','Close','Volume']]
                    df.to_csv(CSV_FILE)
                    st.session_state.core_memory['last_download'] = datetime.utcnow().isoformat()
                    st.session_state.core_memory['rows'] = len(df)
                    save_memory(st.session_state.core_memory)
                    return df
        except Exception as e:
            st.warning(f"MT5 download failed: {e}")

    # Fallback to yfinance
    ticker = "GBPUSD=X"
    cur = start_dt
    frames = []
    
    with st.spinner(f"Downloading from {start_dt.date()} to {end_dt.date()}..."):
        while cur < end_dt:
            chunk_end = min(end_dt, cur + timedelta(days=30))
            try:
                df_chunk = yf.download(ticker, start=cur.strftime("%Y-%m-%d"), 
                                      end=(chunk_end+timedelta(days=1)).strftime("%Y-%m-%d"), 
                                      interval="5m", progress=False, threads=False)
                if df_chunk is not None and not df_chunk.empty:
                    frames.append(df_chunk)
                    st.write(f"✓ Downloaded {cur.date()} to {chunk_end.date()}: {len(df_chunk)} bars")
            except Exception as e:
                st.warning(f"Failed chunk {cur.date()}: {e}")
            cur = chunk_end + timedelta(days=1)
            time.sleep(0.3)
    
    if frames:
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df.columns = [col.strip().title() if isinstance(col, str) else col for col in df.columns]
        df.to_csv(CSV_FILE)
        st.session_state.core_memory['last_download'] = datetime.utcnow().isoformat()
        st.session_state.core_memory['rows'] = len(df)
        save_memory(st.session_state.core_memory)
        return df
    else:
        st.error("No data available in that range.")
        return pd.DataFrame()

# ----------------------------
# Pattern-based Backtester
# ----------------------------
def backtest_with_pattern_learning(df, save_patterns_flag=True):
    """
    Run backtest and save winning patterns
    """
    df_ind = compute_indicators(df)
    df_ind = df_ind.dropna()
    
    n = len(df_ind)
    buy_patterns = []
    sell_patterns = []
    high_trend_patterns = []
    all_trades = []
    
    cash = 0.0
    
    st.write(f"Running backtest on {n} bars from {df_ind.index[0]} to {df_ind.index[-1]}")
    
    # Simple strategy: Buy when +DI > -DI and ADX > 20, Sell when opposite
    for i in range(n - FUTURE_LOOKAHEAD - 1):
        current_adx = df_ind["adx"].iloc[i]
        current_plus_di = df_ind["+di"].iloc[i]
        current_minus_di = df_ind["-di"].iloc[i]
        
        # Skip if indicators not ready
        if pd.isna(current_adx) or pd.isna(current_plus_di) or pd.isna(current_minus_di):
            continue
        
        # BUY signal
        if current_plus_di > current_minus_di and current_adx > 15:
            entry_idx = i + 1
            if entry_idx >= n:
                continue
                
            entry_price = df_ind["Close"].iloc[entry_idx]
            
            # Look ahead for TP/SL
            profit = None
            exit_idx = None
            max_profit = 0
            
            for j in range(entry_idx + 1, min(entry_idx + FUTURE_LOOKAHEAD + 1, n)):
                current_price = df_ind["Close"].iloc[j]
                current_profit = current_price - entry_price
                max_profit = max(max_profit, current_profit)
                
                # TP hit
                if current_profit >= TP_POINTS:
                    profit = current_profit
                    exit_idx = j
                    break
                # SL hit
                if current_profit <= -SL_POINTS:
                    profit = current_profit
                    exit_idx = j
                    break
            
            # If no TP/SL hit, close at end of lookahead
            if profit is None:
                exit_idx = min(entry_idx + FUTURE_LOOKAHEAD, n - 1)
                profit = df_ind["Close"].iloc[exit_idx] - entry_price
            
            cash += profit
            
            trade_info = {
                "type": "BUY",
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_price": float(entry_price),
                "exit_price": float(df_ind["Close"].iloc[exit_idx]),
                "profit": float(profit),
                "max_profit": float(max_profit),
                "adx": float(current_adx),
                "duration": int(exit_idx - entry_idx)
            }
            all_trades.append(trade_info)
            
            # Save winning patterns
            if save_patterns_flag and profit > MIN_PROFIT_TO_SAVE:
                pattern = extract_pattern_features(df_ind, i)
                pattern["profit"] = float(profit)
                pattern["max_profit"] = float(max_profit)
                pattern["trade_type"] = "BUY"
                pattern["duration"] = int(exit_idx - entry_idx)
                
                buy_patterns.append(pattern)
                
                # High trend pattern
                if current_adx > HIGH_TREND_ADX_THRESHOLD:
                    pattern_ht = pattern.copy()
                    pattern_ht["trend_strength"] = "HIGH"
                    high_trend_patterns.append(pattern_ht)
        
        # SELL signal
        elif current_minus_di > current_plus_di and current_adx > 15:
            entry_idx = i + 1
            if entry_idx >= n:
                continue
                
            entry_price = df_ind["Close"].iloc[entry_idx]
            
            profit = None
            exit_idx = None
            max_profit = 0
            
            for j in range(entry_idx + 1, min(entry_idx + FUTURE_LOOKAHEAD + 1, n)):
                current_price = df_ind["Close"].iloc[j]
                current_profit = entry_price - current_price  # Reverse for sell
                max_profit = max(max_profit, current_profit)
                
                if current_profit >= TP_POINTS:
                    profit = current_profit
                    exit_idx = j
                    break
                if current_profit <= -SL_POINTS:
                    profit = current_profit
                    exit_idx = j
                    break
            
            if profit is None:
                exit_idx = min(entry_idx + FUTURE_LOOKAHEAD, n - 1)
                profit = entry_price - df_ind["Close"].iloc[exit_idx]
            
            cash += profit
            
            trade_info = {
                "type": "SELL",
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_price": float(entry_price),
                "exit_price": float(df_ind["Close"].iloc[exit_idx]),
                "profit": float(profit),
                "max_profit": float(max_profit),
                "adx": float(current_adx),
                "duration": int(exit_idx - entry_idx)
            }
            all_trades.append(trade_info)
            
            if save_patterns_flag and profit > MIN_PROFIT_TO_SAVE:
                pattern = extract_pattern_features(df_ind, i)
                pattern["profit"] = float(profit)
                pattern["max_profit"] = float(max_profit)
                pattern["trade_type"] = "SELL"
                pattern["duration"] = int(exit_idx - entry_idx)
                
                sell_patterns.append(pattern)
                
                if current_adx > HIGH_TREND_ADX_THRESHOLD:
                    pattern_ht = pattern.copy()
                    pattern_ht["trend_strength"] = "HIGH"
                    high_trend_patterns.append(pattern_ht)
    
    # Save patterns
    if save_patterns_flag:
        save_patterns(buy_patterns, BUY_PATTERNS_FILE)
        save_patterns(sell_patterns, SELL_PATTERNS_FILE)
        save_patterns(high_trend_patterns, HIGH_TREND_FILE)
        
        st.session_state.core_memory["patterns_count"] = {
            "buy": len(buy_patterns),
            "sell": len(sell_patterns),
            "high_trend": len(high_trend_patterns)
        }
        save_memory(st.session_state.core_memory)
    
    # Calculate metrics
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t["profit"] > 0)
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    winning_trades = [t for t in all_trades if t["profit"] > 0]
    losing_trades = [t for t in all_trades if t["profit"] <= 0]
    
    avg_win = np.mean([t["profit"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["profit"] for t in losing_trades]) if losing_trades else 0
    
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_profit": float(cash),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "buy_patterns_saved": len(buy_patterns),
        "sell_patterns_saved": len(sell_patterns),
        "high_trend_patterns_saved": len(high_trend_patterns),
        "all_trades": all_trades
    }

# ----------------------------
# Pattern-based prediction
# ----------------------------
def predict_with_patterns(df):
    """Predict using saved patterns"""
    df_ind = compute_indicators(df.tail(200))
    df_ind = df_ind.dropna()
    
    if len(df_ind) == 0:
        return {"error": "Not enough data"}
    
    # Get current pattern
    current_pattern = extract_pattern_features(df_ind, -1)
    
    # Load patterns
    buy_patterns = load_patterns(BUY_PATTERNS_FILE)
    sell_patterns = load_patterns(SELL_PATTERNS_FILE)
    high_trend_patterns = load_patterns(HIGH_TREND_FILE)
    
    # Find similar patterns
    similar_buys = find_similar_patterns(current_pattern, buy_patterns, top_k=5, min_similarity=0.65)
    similar_sells = find_similar_patterns(current_pattern, sell_patterns, top_k=5, min_similarity=0.65)
    similar_ht = find_similar_patterns(current_pattern, high_trend_patterns, top_k=5, min_similarity=0.7)
    
    # Calculate confidence scores
    buy_confidence = np.mean([sim for _, sim in similar_buys]) if similar_buys else 0
    sell_confidence = np.mean([sim for _, sim in similar_sells]) if similar_sells else 0
    ht_confidence = np.mean([sim for _, sim in similar_ht]) if similar_ht else 0
    
    # Recommendation
    if buy_confidence > sell_confidence and buy_confidence > 0.7:
        recommendation = "BUY"
    elif sell_confidence > buy_confidence and sell_confidence > 0.7:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    return {
        "recommendation": recommendation,
        "buy_confidence": float(buy_confidence),
        "sell_confidence": float(sell_confidence),
        "high_trend_confidence": float(ht_confidence),
        "similar_buy_patterns": len(similar_buys),
        "similar_sell_patterns": len(similar_sells),
        "similar_ht_patterns": len(similar_ht),
        "current_indicators": current_pattern,
        "top_similar_buys": [(p["timestamp"], p["profit"], sim) for p, sim in similar_buys[:3]],
        "top_similar_sells": [(p["timestamp"], p["profit"], sim) for p, sim in similar_sells[:3]]
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 Pattern Learning FX Bot — GBPUSD M5")
st.markdown("**Self-learning bot that captures and reuses winning trade patterns**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    start = st.date_input("Start date", value=datetime(2020,1,1))
    end = st.date_input("End date", value=datetime.utcnow().date())
    
    st.markdown("---")
    download_btn = st.button("📥 Download Data")
    backtest_btn = st.button("🔬 Run Backtest & Learn Patterns")
    predict_btn = st.button("🎯 Predict with Patterns")
    
    st.markdown("---")
    st.subheader("📊 Pattern Library")
    st.metric("Buy Patterns", st.session_state.core_memory.get("patterns_count", {}).get("buy", 0))
    st.metric("Sell Patterns", st.session_state.core_memory.get("patterns_count", {}).get("sell", 0))
    st.metric("High Trend Patterns", st.session_state.core_memory.get("patterns_count", {}).get("high_trend", 0))
    
    st.markdown("---")
    st.write(f"⏱️ Auto-refresh in: {int(AUTO_REFRESH_SECONDS - elapsed)}s")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Data & Chart")
    
    if download_btn:
        df = download_gbpusd_m5(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            st.success(f"✅ Downloaded {len(df)} bars")
    else:
        if CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
            st.info(f"📊 Loaded {len(df)} bars from local CSV")
        else:
            df = pd.DataFrame()
            st.warning("⚠️ No local data. Click Download to fetch data.")
    
    if not df.empty:
        st.line_chart(df["Close"].tail(1000), height=300)
        
        # Show latest indicators
        try:
            df_ind = compute_indicators(df.tail(100))
            st.subheader("Latest Indicators")
            latest = df_ind.iloc[-1]
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("ADX", f"{latest['adx']:.1f}")
            col_b.metric("+DI", f"{latest['+di']:.1f}")
            col_c.metric("-DI", f"{latest['-di']:.1f}")
            col_d.metric("CCI", f"{latest['cci']:.1f}")
            
            col_e, col_f, col_g, col_h = st.columns(4)
            col_e.metric("SMA9", f"{latest['sma9']:.5f}")
            col_f.metric("EMA30", f"{latest['ema30']:.5f}")
            col_g.metric("ATR", f"{latest['atr']:.5f}")
            col_h.metric("BB Pos", f"{latest['bb_position']:.2f}")
        except Exception as e:
            st.error(f"Error computing indicators: {e}")

with col2:
    st.header("🎯 Predictions")
    
    if predict_btn:
        if not CSV_FILE.exists():
            st.warning("No data file. Download first.")
        else:
            df_local = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
            
            # Check if patterns exist
            if not BUY_PATTERNS_FILE.exists() or not SELL_PATTERNS_FILE.exists():
                st.warning("⚠️ No patterns found. Run backtest first to learn patterns.")
            else:
                with st.spinner("Analyzing patterns..."):
                    result = predict_with_patterns(df_local)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Display recommendation
                        rec = result["recommendation"]
                        if rec == "BUY":
                            st.success(f"🟢 **{rec}** Signal")
                        elif rec == "SELL":
                            st.error(f"🔴 **{rec}** Signal")
                        else:
                            st.info(f"⚪ **{rec}**")
                        
                        # Confidence scores
                        st.metric("Buy Confidence", f"{result['buy_confidence']*100:.1f}%")
                        st.metric("Sell Confidence", f"{result['sell_confidence']*100:.1f}%")
                        st.metric("High Trend Confidence", f"{result['high_trend_confidence']*100:.1f}%")
                        
                        st.markdown("---")
                        st.write(f"📊 Similar Buy Patterns: {result['similar_buy_patterns']}")
                        st.write(f"📊 Similar Sell Patterns: {result['similar_sell_patterns']}")
                        
                        # Show top similar patterns
                        if result['top_similar_buys']:
                            st.markdown("**Top Similar Buy Patterns:**")
                            for ts, profit, sim in result['top_similar_buys']:
                                st.write(f"  - {ts}: Profit={profit:.6f}, Similarity={sim:.2f}")
                        
                        if result['top_similar_sells']:
                            st.markdown("**Top Similar Sell Patterns:**")
                            for ts, profit, sim in result['top_similar_sells']:
                                st.write(f"  - {ts}: Profit={profit:.6f}, Similarity={sim:.2f}")

# Backtest results section
st.markdown("---")
st.header("📊 Backtest Results")

if backtest_btn:
    if not CSV_FILE.exists():
        st.error("No data file. Download first.")
    else:
        df_local = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        
        with st.spinner("Running backtest and learning patterns..."):
            results = backtest_with_pattern_learning(df_local, save_patterns_flag=True)
            
            st.success("✅ Backtest complete!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", results["total_trades"])
            col2.metric("Win Rate", f"{results['win_rate']*100:.1f}%")
            col3.metric("Net Profit", f"{results['net_profit']:.6f}")
            col4.metric("Wins/Losses", f"{results['wins']}/{results['losses']}")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Avg Win", f"{results['avg_win']:.6f}")
            col6.metric("Avg Loss", f"{results['avg_loss']:.6f}")
            col7.metric("Buy Patterns", results["buy_patterns_saved"])
            col8.metric("Sell Patterns", results["sell_patterns_saved"])
            
            st.metric("🔥 High Trend Patterns Saved", results["high_trend_patterns_saved"])
            
            # Show trade details
            st.markdown("---")
            st.subheader("Recent Trades (Last 20)")
            
            trades_df = pd.DataFrame(results["all_trades"][-20:])
            if not trades_df.empty:
                # Format for display
                trades_df["profit"] = trades_df["profit"].apply(lambda x: f"{x:.6f}")
                trades_df["entry_price"] = trades_df["entry_price"].apply(lambda x: f"{x:.5f}")
                trades_df["exit_price"] = trades_df["exit_price"].apply(lambda x: f"{x:.5f}")
                trades_df["adx"] = trades_df["adx"].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(trades_df, use_container_width=True)
            
            # Show winning trades breakdown
            st.markdown("---")
            st.subheader("Winning Trades Analysis")
            
            winning_trades = [t for t in results["all_trades"] if t["profit"] > 0]
            
            if winning_trades:
                buy_wins = [t for t in winning_trades if t["type"] == "BUY"]
                sell_wins = [t for t in winning_trades if t["type"] == "SELL"]
                high_trend_wins = [t for t in winning_trades if t["adx"] > HIGH_TREND_ADX_THRESHOLD]
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Buy Wins", len(buy_wins))
                col_b.metric("Sell Wins", len(sell_wins))
                col_c.metric("High Trend Wins", len(high_trend_wins))
                
                # Profit distribution
                profits = [t["profit"] for t in winning_trades]
                st.write(f"**Profit Range:** {min(profits):.6f} to {max(profits):.6f}")
                st.write(f"**Median Profit:** {np.median(profits):.6f}")
                
                # Show best trades
                st.markdown("**Top 10 Most Profitable Trades:**")
                top_trades = sorted(winning_trades, key=lambda x: x["profit"], reverse=True)[:10]
                top_df = pd.DataFrame(top_trades)
                top_df["profit"] = top_df["profit"].apply(lambda x: f"{x:.6f}")
                top_df["adx"] = top_df["adx"].apply(lambda x: f"{x:.1f}")
                st.dataframe(top_df[["type", "profit", "adx", "duration"]], use_container_width=True)

# Pattern library viewer
st.markdown("---")
st.header("📚 Pattern Library Viewer")

view_patterns = st.selectbox("View Patterns", ["None", "Buy Patterns", "Sell Patterns", "High Trend Patterns"])

if view_patterns != "None":
    if view_patterns == "Buy Patterns":
        patterns = load_patterns(BUY_PATTERNS_FILE)
        st.write(f"**Total Buy Patterns:** {len(patterns)}")
    elif view_patterns == "Sell Patterns":
        patterns = load_patterns(SELL_PATTERNS_FILE)
        st.write(f"**Total Sell Patterns:** {len(patterns)}")
    else:
        patterns = load_patterns(HIGH_TREND_FILE)
        st.write(f"**Total High Trend Patterns:** {len(patterns)}")
    
    if patterns:
        # Show sample patterns
        st.write("**Sample Patterns (first 10):**")
        for i, pattern in enumerate(patterns[:10]):
            with st.expander(f"Pattern {i+1}: Profit={pattern.get('profit', 0):.6f}, Type={pattern.get('trade_type', 'N/A')}"):
                st.json(pattern)
        
        # Statistics
        st.markdown("---")
        st.subheader("Pattern Statistics")
        
        profits = [p.get("profit", 0) for p in patterns]
        adx_values = [p.get("adx", 0) for p in patterns]
        durations = [p.get("duration", 0) for p in patterns]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Profit", f"{np.mean(profits):.6f}")
        col1.metric("Max Profit", f"{max(profits):.6f}")
        
        col2.metric("Avg ADX", f"{np.mean(adx_values):.1f}")
        col2.metric("Max ADX", f"{max(adx_values):.1f}")
        
        col3.metric("Avg Duration", f"{np.mean(durations):.1f} bars")
        col3.metric("Max Duration", f"{max(durations)} bars")

# Pattern matching test
st.markdown("---")
st.header("🔍 Pattern Matching Test")

if st.button("Test Pattern Matching on Latest Bar"):
    if not CSV_FILE.exists():
        st.warning("No data file.")
    else:
        df_local = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        df_ind = compute_indicators(df_local.tail(100))
        df_ind = df_ind.dropna()
        
        if len(df_ind) > 0:
            current = extract_pattern_features(df_ind, -1)
            
            st.write("**Current Market Pattern:**")
            st.json(current)
            
            # Test against all pattern types
            buy_patterns = load_patterns(BUY_PATTERNS_FILE)
            sell_patterns = load_patterns(SELL_PATTERNS_FILE)
            ht_patterns = load_patterns(HIGH_TREND_FILE)
            
            if buy_patterns or sell_patterns or ht_patterns:
                st.markdown("---")
                st.subheader("Similar Patterns Found:")
                
                if buy_patterns:
                    similar_buys = find_similar_patterns(current, buy_patterns, top_k=3, min_similarity=0.6)
                    if similar_buys:
                        st.write("**Buy Patterns:**")
                        for pattern, similarity in similar_buys:
                            st.write(f"  - Similarity: {similarity:.2f}, Profit: {pattern['profit']:.6f}, Time: {pattern['timestamp']}")
                
                if sell_patterns:
                    similar_sells = find_similar_patterns(current, sell_patterns, top_k=3, min_similarity=0.6)
                    if similar_sells:
                        st.write("**Sell Patterns:**")
                        for pattern, similarity in similar_sells:
                            st.write(f"  - Similarity: {similarity:.2f}, Profit: {pattern['profit']:.6f}, Time: {pattern['timestamp']}")
                
                if ht_patterns:
                    similar_ht = find_similar_patterns(current, ht_patterns, top_k=3, min_similarity=0.65)
                    if similar_ht:
                        st.write("**High Trend Patterns:**")
                        for pattern, similarity in similar_ht:
                            st.write(f"  - Similarity: {similarity:.2f}, Profit: {pattern['profit']:.6f}, Time: {pattern['timestamp']}")
            else:
                st.info("No patterns in library yet. Run backtest first.")

# Summary footer
st.markdown("---")
st.subheader("💾 System Summary")

col_x, col_y, col_z = st.columns(3)
col_x.write(f"**Last Download:** {st.session_state.core_memory.get('last_download', 'Never')}")
col_y.write(f"**Total Bars:** {st.session_state.core_memory.get('rows', 0)}")
col_z.write(f"**Data File:** {'✅ Exists' if CSV_FILE.exists() else '❌ Missing'}")

st.markdown("---")
st.info("""
**How it works:**
1. **Download Data**: Get GBPUSD M5 data from 2020 to present
2. **Run Backtest**: The bot trades using ADX/DI strategy and saves all winning patterns (profit > 5 pips)
3. **Pattern Learning**: Stores indicator values at entry for winning trades - categorized as Buy, Sell, and High Trend
4. **Predict**: Compares current market state to stored patterns using similarity scoring
5. **Forward Testing**: Use learned patterns to make trading decisions

**Indicators Used:**
- SMA 9, EMA 30
- CCI Period 20
- ADX Period 8 with +DI, -DI
- Bollinger Bands
- ATR Period 14
""")
