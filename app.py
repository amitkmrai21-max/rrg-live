import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai

try:
    import talib
except Exception:
    talib = None

st.set_page_config(
    page_title="Smart Trade AI Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    .stDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    @media (max-width: 600px) {
        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
        h1 {
            font-size: 1.7rem !important;
        }
        h2, h3 {
            font-size: 1.25rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = "trade_data.json"
DEFAULT_BALANCE = 100000.0

CRYPTO_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
]

RRG_SYMBOLS = [
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
    "DOGE-USD",
]

RRG_BENCHMARK = "BTC-USD"


def load_trade_data():
    if not os.path.exists(DATA_FILE):
        return {"balance": DEFAULT_BALANCE, "positions": [], "history": []}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "balance": float(data.get("balance", DEFAULT_BALANCE)),
            "positions": data.get("positions", []),
            "history": data.get("history", []),
        }
    except Exception:
        return {"balance": DEFAULT_BALANCE, "positions": [], "history": []}


def save_trade_data():
    try:
        payload = {
            "balance": st.session_state.balance,
            "positions": st.session_state.positions,
            "history": st.session_state.trade_history,
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


saved_data = load_trade_data()

if "balance" not in st.session_state:
    st.session_state.balance = saved_data["balance"]
if "positions" not in st.session_state:
    st.session_state.positions = saved_data["positions"]
if "trade_history" not in st.session_state:
    st.session_state.trade_history = saved_data["history"]
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

client = None
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if api_key:
        client = genai.Client(api_key=api_key)
except Exception:
    client = None


def symbol_to_yahoo(symbol):
    return symbol.replace("USDT", "-USD")


def money(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "₹0.00"


def safe_float(value, default=None):
    try:
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return default
        return num
    except Exception:
        return default


@st.cache_data(ttl=60, show_spinner=False)
def get_crypto_data(symbol, period="3mo", interval="1d"):
    ticker = symbol_to_yahoo(symbol)
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if "Close" not in data.columns:
            return None

        data = data.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["Close"])

        close = data["Close"]
        data["SMA20"] = close.rolling(20, min_periods=20).mean()
        data["SMA50"] = close.rolling(50, min_periods=50).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        data["RSI"] = 100 - (100 / (1 + rs))

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
        return data
    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_live_prices(symbols_tuple):
    tickers = [symbol_to_yahoo(s) for s in symbols_tuple]
    prices = {s: None for s in symbols_tuple}
    try:
        data = yf.download(
            tickers=tickers,
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker",
            threads=True,
            auto_adjust=False,
        )
        for symbol, ticker in zip(symbols_tuple, tickers):
            try:
                if len(tickers) > 1:
                    close = data[ticker]["Close"]
                else:
                    close = data["Close"]
                close = close.dropna()
                if not close.empty:
                    prices[symbol] = float(close.iloc[-1])
            except Exception:
                continue
    except Exception:
        pass
    return prices


def get_current_price(symbol):
    prices = fetch_live_prices(tuple(CRYPTO_SYMBOLS))
    return prices.get(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def download_rrg_data(tickers):
    try:
        return yf.download(
            tickers=list(tickers),
            period="90d",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception:
        return None


def extract_close(data, ticker):
    if data is None:
        return pd.Series(dtype=float)
    try:
        return data[ticker]["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


def detect_swing_points(df, window=5):
    df = df.copy()
    df["Swing_High"] = False
    df["Swing_Low"] = False
    if len(df) < window * 2 + 1:
        return df

    for i in range(window, len(df) - window):
        left_max = df["High"].iloc[i - window:i].max()
        right_max = df["High"].iloc[i + 1:i + window + 1].max()
        if df["High"].iloc[i] > left_max and df["High"].iloc[i] > right_max:
            df.loc[df.index[i], "Swing_High"] = True

        left_min = df["Low"].iloc[i - window:i].min()
        right_min = df["Low"].iloc[i + 1:i + window + 1].min()
        if df["Low"].iloc[i] < left_min and df["Low"].iloc[i] < right_min:
            df.loc[df.index[i], "Swing_Low"] = True

    return df


def detect_market_structure(df):
    df = detect_swing_points(df, window=5)
    swings = df[(df["Swing_High"]) | (df["Swing_Low"])].tail(10)

    if len(swings) < 4:
        return "NEUTRAL", None, None

    highs = swings[swings["Swing_High"]]["High"].tolist()
    lows = swings[swings["Swing_Low"]]["Low"].tolist()

    trend = "NEUTRAL"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            trend = "BULLISH"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            trend = "BEARISH"

    last_swing_high = swings[swings["Swing_High"]]["High"].iloc[-1] if len(swings[swings["Swing_High"]]) > 0 else None
    last_swing_low = swings[swings["Swing_Low"]]["Low"].iloc[-1] if len(swings[swings["Swing_Low"]]) > 0 else None
    current_price = df["Close"].iloc[-1]

    bos = None
    if trend == "BULLISH" and last_swing_high and current_price > last_swing_high:
        bos = "BOS_UP"
    elif trend == "BEARISH" and last_swing_low and current_price < last_swing_low:
        bos = "BOS_DOWN"

    choch = None
    if trend == "BULLISH" and last_swing_low and current_price < last_swing_low:
        choch = "ChoCh_BEARISH"
    elif trend == "BEARISH" and last_swing_high and current_price > last_swing_high:
        choch = "ChoCh_BULLISH"

    return trend, bos, choch


def detect_liquidity_sweeps(df):
    df = detect_swing_points(df, window=5)
    sweeps = []
    for i in range(1, len(df)):
        if bool(df["Swing_High"].iloc[i - 1]):
            prev_high = df["High"].iloc[i - 1]
            if df["High"].iloc[i] > prev_high and df["Close"].iloc[i] < prev_high:
                sweeps.append(("BEARISH_SWEEP", prev_high, df.index[i]))
        if bool(df["Swing_Low"].iloc[i - 1]):
            prev_low = df["Low"].iloc[i - 1]
            if df["Low"].iloc[i] < prev_low and df["Close"].iloc[i] > prev_low:
                sweeps.append(("BULLISH_SWEEP", prev_low, df.index[i]))
    return sweeps[-3:] if sweeps else []


def detect_fvg(df):
    fvg_list = []
    if len(df) < 3:
        return fvg_list

    for i in range(2, len(df)):
        first_high = df["High"].iloc[i - 2]
        first_low = df["Low"].iloc[i - 2]
        third_low = df["Low"].iloc[i]
        third_high = df["High"].iloc[i]
        middle_open = df["Open"].iloc[i - 1]
        middle_close = df["Close"].iloc[i - 1]

        prev_bodies = (df["Close"].iloc[max(0, i - 10):i - 1] - df["Open"].iloc[max(0, i - 10):i - 1]).abs()
        avg_body_size = prev_bodies.mean() if len(prev_bodies) > 0 else 0.0
        middle_body = abs(middle_close - middle_open)

        if third_low > first_high and middle_body > avg_body_size * 1.5:
            fvg_list.append(("BULLISH_FVG", first_high, third_low, i))
        elif third_high < first_low and middle_body > avg_body_size * 1.5:
            fvg_list.append(("BEARISH_FVG", first_low, third_high, i))

    return fvg_list[-3:] if fvg_list else []


def detect_order_blocks(df):
    ob_list = []
    if len(df) < 2:
        return ob_list

    for i in range(1, len(df)):
        prev_candle = df.iloc[i - 1]
        curr_candle = df.iloc[i]
        if prev_candle["Close"] < prev_candle["Open"] and curr_candle["Close"] > curr_candle["Open"]:
            if curr_candle["Close"] > prev_candle["High"]:
                ob_list.append(("BULLISH_OB", prev_candle["Low"], prev_candle["High"], i))
        if prev_candle["Close"] > prev_candle["Open"] and curr_candle["Close"] < curr_candle["Open"]:
            if curr_candle["Close"] < prev_candle["Low"]:
                ob_list.append(("BEARISH_OB", prev_candle["High"], prev_candle["Low"], i))
    return ob_list[-3:] if ob_list else []


def detect_divergence(df):
    if "RSI" not in df.columns or len(df) < 30:
        return "NONE"

    recent = df.tail(20).copy()
    price_lows = recent["Low"].tolist()
    rsi_lows = recent["RSI"].tolist()
    price_highs = recent["High"].tolist()
    rsi_highs = recent["RSI"].tolist()

    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        if price_lows[-1] < price_lows[-2] and rsi_lows[-1] > rsi_lows[-2]:
            return "BULLISH_DIV"

    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        if price_highs[-1] > price_highs[-2] and rsi_highs[-1] < rsi_highs[-2]:
            return "BEARISH_DIV"

    return "NONE"


def analyze_volume(df):
    if "Volume" not in df.columns:
        return "NEUTRAL", 0.0, "NEUTRAL"

    vol = df["Volume"]
    vol_sma20 = vol.rolling(20, min_periods=20).mean()
    vol_ratio = vol / vol_sma20.replace(0, np.nan)
    latest_ratio = float(vol_ratio.iloc[-1]) if len(vol_ratio) > 0 and pd.notna(vol_ratio.iloc[-1]) else 1.0

    spike = "NONE"
    if latest_ratio > 2.5:
        spike = "STRONG_SPIKE"
    elif latest_ratio > 1.8:
        spike = "MODERATE_SPIKE"
    elif latest_ratio < 0.6:
        spike = "LOW_VOLUME"

    vol_trend = "NEUTRAL"
    if len(vol_sma20) > 20 and pd.notna(vol_sma20.iloc[-1]) and pd.notna(vol_sma20.iloc[-5]):
        if vol_sma20.iloc[-1] > vol_sma20.iloc[-5]:
            vol_trend = "INCREASING"
        elif vol_sma20.iloc[-1] < vol_sma20.iloc[-5]:
            vol_trend = "DECREASING"

    return spike, latest_ratio, vol_trend


def calculate_volume_profile(df, num_candles=30):
    if len(df) < num_candles or "Volume" not in df.columns:
        return None, None, None

    sliced = df.iloc[-num_candles:].copy()
    low = float(sliced["Low"].min())
    high = float(sliced["High"].max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None, None, None

    bins = np.linspace(low, high, 50)
    profile = {}

    for i in range(len(bins) - 1):
        mask = (sliced["Close"] >= bins[i]) & (sliced["Close"] < bins[i + 1])
        profile[bins[i]] = float(sliced.loc[mask, "Volume"].sum())

    if not profile:
        return None, None, None

    poc_price = max(profile, key=profile.get)
    total_volume = sum(profile.values())
    sorted_profile = sorted(profile.items(), key=lambda x: x[1], reverse=True)

    cumulative = 0.0
    vah = poc_price
    val = poc_price

    for price, vol in sorted_profile:
        cumulative += vol
        if cumulative <= total_volume * 0.7:
            vah = max(vah, price)
            val = min(val, price)

    return poc_price, vah, val


def detect_candlestick_patterns(df):
    patterns = {}
    if talib is None or len(df) < 10:
        return "NEUTRAL", patterns

    try:
        patterns["Hammer"] = talib.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1]
        patterns["Engulfing"] = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1]
        patterns["Harami"] = talib.CDLHARAMI(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1]
        patterns["Morning_Star"] = talib.CDLMORNINGSTAR(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1]
        patterns["Evening_Star"] = talib.CDLEVENINGSTAR(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1]
        patterns["Doji"] = talib.CDLDOJI(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1]
    except Exception:
        return "NEUTRAL", patterns

    bullish = sum(1 for v in patterns.values() if v > 0)
    bearish = sum(1 for v in patterns.values() if v < 0)

    if bullish > bearish:
        return "BULLISH", patterns
    if bearish > bullish:
        return "BEARISH", patterns
    return "NEUTRAL", patterns


@st.cache_data(ttl=120, show_spinner=False)
def get_multi_timeframe_data(symbol):
    data_1d = get_crypto_data(symbol, period="3mo", interval="1d")
    data_4h = get_crypto_data(symbol, period="1mo", interval="1h")
    return data_1d, data_4h


def check_confluence(data_1d, data_4h):
    if data_1d is None or data_4h is None:
        return "NO_DATA", 0

    trend_1d, _, _ = detect_market_structure(data_1d)
    trend_4h, _, _ = detect_market_structure(data_4h)

    if trend_1d == trend_4h and trend_1d != "NEUTRAL":
        return "STRONG", 100
    if trend_1d == "NEUTRAL" or trend_4h == "NEUTRAL":
        return "MODERATE", 50
    return "WEAK", 0


def calculate_signal_score(
    trend,
    bos,
    choch,
    sweeps,
    fvg,
    ob,
    divergence,
    vol_spike,
    vol_ratio,
    candle_signal,
    confluence_score,
    rsi,
    sma_signal,
):
    score = 0
    reasons = []

    if trend == "BULLISH":
        score += 12
        reasons.append("Bullish structure")
    elif trend == "BEARISH":
        score -= 12
        reasons.append("Bearish structure")

    if bos == "BOS_UP":
        score += 13
        reasons.append("BOS up")
    elif bos == "BOS_DOWN":
        score -= 13
        reasons.append("BOS down")

    if choch == "ChoCh_BULLISH":
        score += 10
        reasons.append("ChoCh bullish")
    elif choch == "ChoCh_BEARISH":
        score -= 10
        reasons.append("ChoCh bearish")

    if sweeps:
        latest_sweep = sweeps[-1]
        if latest_sweep[0] == "BULLISH_SWEEP":
            score += 15
            reasons.append(f"Bullish sweep @ {latest_sweep[1]:.2f}")
        elif latest_sweep[0] == "BEARISH_SWEEP":
            score -= 15
            reasons.append(f"Bearish sweep @ {latest_sweep[1]:.2f}")

    if fvg:
        latest_fvg = fvg[-1]
        if latest_fvg[0] == "BULLISH_FVG":
            score += 15
            reasons.append("Bullish FVG")
        elif latest_fvg[0] == "BEARISH_FVG":
            score -= 15
            reasons.append("Bearish FVG")

    if ob:
        latest_ob = ob[-1]
        if latest_ob[0] == "BULLISH_OB":
            score += 10
            reasons.append("Bullish OB")
        elif latest_ob[0] == "BEARISH_OB":
            score -= 10
            reasons.append("Bearish OB")

    if divergence == "BULLISH_DIV":
        score += 15
        reasons.append("Bullish divergence")
    elif divergence == "BEARISH_DIV":
        score -= 15
        reasons.append("Bearish divergence")

    if vol_spike == "STRONG_SPIKE":
        score += 15
        reasons.append(f"Volume spike {vol_ratio:.2f}x")
    elif vol_spike == "MODERATE_SPIKE":
        score += 10
        reasons.append(f"Volume moderate {vol_ratio:.2f}x")
    elif vol_spike == "LOW_VOLUME":
        score -= 10
        reasons.append("Low volume")

    if candle_signal == "BULLISH":
        score += 10
        reasons.append("Bullish candles")
    elif candle_signal == "BEARISH":
        score -= 10
        reasons.append("Bearish candles")

    score += (confluence_score / 100) * 10
    if confluence_score == 100:
        reasons.append("Strong confluence")
    elif confluence_score == 50:
        reasons.append("Moderate confluence")

    if sma_signal == "BUY" and rsi < 65:
        score += 5
        reasons.append("RSI + SMA bullish")
    elif sma_signal == "SELL" and rsi > 35:
        score -= 5
        reasons.append("RSI + SMA bearish")

    final_score = max(0, min(100, 50 + score))
    return final_score, reasons


def generate_ai_signal_enriched(
    pair,
    timeframe,
    price,
    rsi,
    sma20,
    sma50,
    trend,
    bos,
    choch,
    sweeps,
    fvg,
    ob,
    divergence,
    vol_spike,
    vol_ratio,
    candle_signal,
    confluence,
    confluence_score,
):
    if not client:
        return None

    sweep_text = ", ".join([f"{s[0]}@{s[1]:.2f}" for s in sweeps[-2:]]) if sweeps else "None"
    fvg_text = ", ".join([f[0] for f in fvg[-2:]]) if fvg else "None"
    ob_text = ", ".join([o[0] for o in ob[-2:]]) if ob else "None"

    prompt = f"""You are a professional crypto technical analyst with expertise in Smart Money Concepts (SMC), order flow, and institutional trading.

Analyze this ENRICHED market data:

Pair: {pair}
Timeframe: {timeframe}
Current Price: ${price:.6f}

Classic Indicators:
RSI: {rsi:.2f}
SMA20: ${sma20:.6f}
SMA50: ${sma50:.6f}

Market Structure:
Trend: {trend}
BOS: {bos if bos else "None"}
ChoCh: {choch if choch else "None"}

Smart Money Concepts:
Liquidity Sweeps: {sweep_text}
Fair Value Gaps: {fvg_text}
Order Blocks: {ob_text}
RSI Divergence: {divergence}

Volume Analysis:
Volume Spike: {vol_spike}
Volume Ratio: {vol_ratio:.2f}x

Candlestick Patterns:
Pattern Signal: {candle_signal}

Multi-Timeframe Confluence:
Confluence: {confluence}
Confluence Score: {confluence_score}/100

Your Task:
Give exactly one final action: BUY, SELL, or WAIT.
Consider ALL factors above before deciding.
If data is mixed, choose WAIT.

Reply strictly in this format:

### ACTION: [BUY / SELL / WAIT]
- Entry: [price or range]
- Target: [price]
- Stop-Loss: [price]
- Confidence: [0-100]%
- Reasoning: [short explanation in Hinglish]
"""

    for model_name in ["gemini-flash-lite-latest", "gemini-flash-latest"]:
        try:
            result = client.models.generate_content(model=model_name, contents=prompt)
            if result.text:
                return result.text.strip()[:6000]
        except Exception:
            continue

    return None


def generate_fallback_signal(price, rsi, sma20, sma50):
    if sma20 > sma50 and rsi < 68:
        action = "BUY 🟢"
        entry = f"${price:.6f}"
        target = f"${price * 1.02:.6f}"
        stop = f"${price * 0.99:.6f}"
        reason = "SMA20 SMA50 ke upar hai aur RSI overbought zone me nahi hai."
    elif sma20 < sma50 or rsi > 70:
        action = "SELL 🔴"
        entry = f"${price:.6f}"
        target = f"${price * 0.98:.6f}"
        stop = f"${price * 1.01:.6f}"
        reason = "Trend weak hai ya RSI overbought zone me hai."
    else:
        action = "WAIT ⚪"
        entry = f"${price:.6f}"
        target = "N/A"
        stop = "N/A"
        reason = "Market mixed/consolidating hai. Confirmation ka wait karein."

   return (
    f"### ACTION: {action}

"
    f"- **Entry**: {entry}
"
    f"- **Target**: {target}
"
    f"- **Stop-Loss**: {stop}
"
    f"- **Reasoning**: {reason}"
)


def execute_trade(action, symbol, quantity, price):
    if action == "BUY":
        cost = quantity * price
        if cost > st.session_state.balance:
            return False, "Insufficient balance."

        st.session_state.balance -= cost
        st.session_state.positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": price,
                "timestamp": datetime.now().isoformat(),
            }
        )
        st.session_state.trade_history.append(
            {
                "action": "BUY",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().isoformat(),
            }
        )
        save_trade_data()
        return True, "Buy executed."

    if action == "SELL":
        pos_to_close = None
        for pos in st.session_state.positions:
            if pos["symbol"] == symbol:
                pos_to_close = pos
                break

        if not pos_to_close:
            return False, "No open position to sell."

        pnl = (price - pos_to_close["entry_price"]) * pos_to_close["quantity"]
        st.session_state.balance += pos_to_close["quantity"] * price
        st.session_state.positions.remove(pos_to_close)
        st.session_state.trade_history.append(
            {
                "action": "SELL",
                "symbol": symbol,
                "quantity": pos_to_close["quantity"],
                "price": price,
                "pnl": pnl,
                "timestamp": datetime.now().isoformat(),
            }
        )
        save_trade_data()
        return True, f"Sold. P&L: ₹{pnl:,.2f}"

    return False, "Invalid action."


def build_rrg_figure(data, symbols, benchmark):
    benchmark_close = extract_close(data, benchmark)
    if benchmark_close.empty or len(benchmark_close) < 30:
        return None, "Benchmark data unavailable or insufficient."

    series_map = {}
    colors = ["#00CC96", "#EF553B", "#636EFA", "#FFA15A", "#AB63FA", "#19D3F3"]

    for idx, ticker in enumerate(symbols):
        coin_close = extract_close(data, ticker)
        if coin_close.empty:
            continue

        relative_strength = (coin_close / benchmark_close) * 100
        relative_strength = relative_strength.dropna()
        if len(relative_strength) < 30:
            continue

        rs_ratio = (relative_strength / relative_strength.rolling(14, min_periods=14).mean()) * 100
        rs_momentum = (rs_ratio / rs_ratio.shift(1)) * 100

        combined = pd.concat([rs_ratio, rs_momentum], axis=1).dropna()
        if combined.empty:
            continue

        combined.columns = ["x", "y"]
        name = ticker.replace("-USD", "")
        series_map[name] = {
            "frame": combined,
            "color": colors[idx % len(colors)],
        }

    if not series_map:
        return None, "No valid RRG data."

    common_index = None
    for item in series_map.values():
        frame = item["frame"]
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)

    if common_index is None or len(common_index) < 2:
        return None, "No common dates."

    common_index = common_index.sort_values()
    frame_dates = common_index[-20:] if len(common_index) > 20 else common_index
    if len(frame_dates) < 2:
        return None, "Not enough dates for RRG."

    all_x = pd.concat([item["frame"]["x"] for item in series_map.values()])
    all_y = pd.concat([item["frame"]["y"] for item in series_map.values()])
    x_min = min(safe_float(all_x.min(), 98) - 1.5, 98)
    x_max = max(safe_float(all_x.max(), 102) + 1.5, 102)
    y_min = min(safe_float(all_y.min(), 98) - 1.5, 98)
    y_max = max(safe_float(all_y.max(), 102) + 1.5, 102)

    def frame_traces(date_value):
        traces = []
        for name, item in series_map.items():
            frame = item["frame"]
            subset = frame.loc[:date_value].tail(3)
            if subset.empty:
                continue
            color = item["color"]
            dates_text = [date.strftime("%d/%m") for date in subset.index]
            traces.append(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6, color=color),
                    name=name,
                    legendgroup=name,
                    text=dates_text,
                    hovertemplate=(
                        name
                        + "<br>%{text}"
                        + "<br>RS-Ratio: %{x:.2f}"
                        + "<br>RS-Momentum: %{y:.2f}"
                        + "<extra></extra>"
                    ),
                )
            )
            traces.append(
                go.Scatter(
                    x=[subset["x"].iloc[-1]],
                    y=[subset["y"].iloc[-1]],
                    mode="markers",
                    marker=dict(size=13, color=color, line=dict(color="white", width=2)),
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=(
                        name
                        + " latest"
                        + "<br>RS-Ratio: %{x:.2f}"
                        + "<br>RS-Momentum: %{y:.2f}"
                        + "<extra></extra>"
                    ),
                )
            )
        return traces

    figure = go.Figure(data=frame_traces(frame_dates[-1]))
    figure.add_shape(type="line", x0=100, x1=100, y0=y_min, y1=y_max, line=dict(color="gray", dash="dash"))
    figure.add_shape(type="line", x0=x_min, x1=x_max, y0=100, y1=100, line=dict(color="gray", dash="dash"))
    figure.add_shape(type="rect", x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="#2ecc71", opacity=0.10, line_width=0, layer="below")
    figure.add_shape(type="rect", x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="#f39c12", opacity=0.10, line_width=0, layer="below")
    figure.add_shape(type="rect", x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="#e74c3c", opacity=0.10, line_width=0, layer="below")
    figure.add_shape(type="rect", x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="#3498db", opacity=0.10, line_width=0, layer="below")

    figure.update_layout(
        title="Relative Rotation Graph (vs BTC)",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=600,
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure, None


tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart Analyzer", "🔄 Daily RRG", "📝 Paper Trading", "🤖 AI Signals"])

with tab1:
    st.header("Live Chart Analyzer")
    selected = st.selectbox("Select Pair", CRYPTO_SYMBOLS, key="chart_pair")
    data = get_crypto_data(selected, period="3mo", interval="1d")

    if data is not None and not data.empty:
        price = get_current_price(selected) or data["Close"].iloc[-1]
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
            )
        )
        if "SMA20" in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA20", line=dict(color="blue", width=1.5)))
        if "SMA50" in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA50", line=dict(color="orange", width=1.5)))

        fig.update_layout(title=f"{selected} Price & Indicators", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Current Price", f"${price:.6f}")
        if "RSI" in data.columns and pd.notna(data["RSI"].iloc[-1]):
            st.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.2f}")
        if "SMA20" in data.columns and "SMA50" in data.columns:
            st.write(f"**SMA20**: ${data['SMA20'].iloc[-1]:.6f} | **SMA50**: ${data['SMA50'].iloc[-1]:.6f}")
    else:
        st.warning("No data available.")

with tab2:
    st.header("Daily RRG Chart (vs BTC)")
    rrg_data = download_rrg_data(RRG_SYMBOLS + [RRG_BENCHMARK])
    if rrg_data is not None:
        fig_rrg, error = build_rrg_figure(rrg_data, RRG_SYMBOLS, RRG_BENCHMARK)
        if fig_rrg is not None:
            st.plotly_chart(fig_rrg, use_container_width=True)
            st.caption("🟢 Leading | 🟠 Weakening | 🔴 Lagging | 🔵 Improving")
        else:
            st.warning(error or "RRG chart could not be generated.")
    else:
        st.warning("Failed to fetch RRG data.")

with tab3:
    st.header("Live Paper Trading")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Portfolio")
        st.metric("Balance", money(st.session_state.balance))
        if st.session_state.positions:
            st.write("**Open Positions**:")
            for pos in st.session_state.positions:
                current = get_current_price(pos["symbol"]) or pos["entry_price"]
                pnl = (current - pos["entry_price"]) * pos["quantity"]
                st.write(f"- **{pos['symbol']}**: {pos['quantity']} @ ${pos['entry_price']:.6f} → P&L: ₹{pnl:,.2f}")
        else:
            st.info("No open positions.")

    with col2:
        st.subheader("Execute Trade")
        trade_action = st.selectbox("Action", ["BUY", "SELL"], key="trade_action")
        trade_symbol = st.selectbox("Symbol", CRYPTO_SYMBOLS, key="trade_symbol")
        trade_qty = st.number_input("Quantity", min_value=0.001, value=0.01, step=0.001, key="trade_qty")
        default_price = get_current_price(trade_symbol) or 1.0
        trade_price = st.number_input("Price (USD)", min_value=0.0001, value=float(default_price), step=0.0001, key="trade_price")
        if st.button("Execute Trade", key="exec_trade"):
            success, msg = execute_trade(trade_action, trade_symbol, trade_qty, trade_price)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.subheader("Trade History")
        if st.session_state.trade_history:
            for h in reversed(st.session_state.trade_history[-10:]):
                st.write(f"{h['timestamp'][:19]} | {h['action']} {h['symbol']} @ ${h['price']:.6f}")
                if "pnl" in h:
                    st.write(f"  → P&L: ₹{h['pnl']:,.2f}")
        else:
            st.info("No trades yet.")

with tab4:
    st.header("🤖 AI Signals Pro")
    signal_pair = st.selectbox("Select Pair for AI Signal", CRYPTO_SYMBOLS, key="signal_pair")

    if st.button("Generate Deep AI Signal", key="gen_signal"):
        with st.spinner("Analyzing market structure, volume, patterns & confluence..."):
            data_1d, data_4h = get_multi_timeframe_data(signal_pair)

            if data_1d is not None and not data_1d.empty:
                price_sig = get_current_price(signal_pair) or data_1d["Close"].iloc[-1]
                rsi_sig = data_1d["RSI"].iloc[-1] if "RSI" in data_1d.columns and pd.notna(data_1d["RSI"].iloc[-1]) else 50.0
                sma20_sig = data_1d["SMA20"].iloc[-1] if "SMA20" in data_1d.columns and pd.notna(data_1d["SMA20"].iloc[-1]) else price_sig
                sma50_sig = data_1d["SMA50"].iloc[-1] if "SMA50" in data_1d.columns and pd.notna(data_1d["SMA50"].iloc[-1]) else price_sig

                trend, bos, choch = detect_market_structure(data_1d)
                sweeps = detect_liquidity_sweeps(data_1d)
                fvg = detect_fvg(data_1d)
                ob = detect_order_blocks(data_1d)
                divergence = detect_divergence(data_1d)
                vol_spike, vol_ratio, vol_trend = analyze_volume(data_1d)
                poc, vah, val = calculate_volume_profile(data_1d)
                candle_signal, patterns = detect_candlestick_patterns(data_1d)
                confluence, confluence_score = check_confluence(data_1d, data_4h)

                sma_signal = "BUY" if sma20_sig > sma50_sig else "SELL"
                score, reasons = calculate_signal_score(
                    trend,
                    bos,
                    choch,
                    sweeps,
                    fvg,
                    ob,
                    divergence,
                    vol_spike,
                    vol_ratio,
                    candle_signal,
                    confluence_score,
                    rsi_sig,
                    sma_signal,
                )

                st.metric("Signal Score (0-100)", f"{score:.0f}")
                st.write("**Key Factors**: " + (", ".join(reasons) if reasons else "None"))

                with st.expander("📊 Smart Money Concepts Details"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write("**Liquidity Sweeps**")
                        if sweeps:
                            for s in sweeps:
                                st.write(f"- {s[0]} @ {s[1]:.2f}")
                        else:
                            st.write("- None")
                    with c2:
                        st.write("**Fair Value Gaps**")
                        if fvg:
                            for item in fvg:
                                st.write(f"- {item[0]}")
                        else:
                            st.write("- None")
                    with c3:
                        st.write("**Order Blocks**")
                        if ob:
                            for item in ob:
                                st.write(f"- {item[0]}")
                        else:
                            st.write("- None")

                    st.write(f"**Divergence**: {divergence}")
                    if poc is not None:
                        st.write(f"**Volume Profile**: POC={poc:.2f}, VAH={vah:.2f}, VAL={val:.2f}")

                    if patterns:
                        st.write("**Candlestick Patterns**")
                        for k, v in patterns.items():
                            st.write(f"- {k}: {v}")

                ai_signal = generate_ai_signal_enriched(
                    signal_pair,
                    "1D",
                    price_sig,
                    rsi_sig,
                    sma20_sig,
                    sma50_sig,
                    trend,
                    bos,
                    choch,
                    sweeps,
                    fvg,
                    ob,
                    divergence,
                    vol_spike,
                    vol_ratio,
                    candle_signal,
                    confluence,
                    confluence_score,
                )

                if not ai_signal:
                    ai_signal = generate_fallback_signal(price_sig, rsi_sig, sma20_sig, sma50_sig)

                st.session_state.last_signal = ai_signal
            else:
                st.error("No data for signal generation.")

    if st.session_state.last_signal:
        st.markdown(st.session_state.last_signal)

if (datetime.now() - st.session_state.last_refresh).total_seconds() > 60:
    st.session_state.last_refresh = datetime.now()
    st.rerun()