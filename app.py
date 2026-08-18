import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# TA-Lib optional rakha hai.
# Agar installed nahi hai to app crash nahi karega.
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    talib = None
    TALIB_AVAILABLE = False

# Gemini
try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Trade AI Pro 🚀",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu, header, footer,
.stAppHeader, .stDeployButton,
div[data-testid="stDecoration"] {
    display: none;
}

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
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "trade_data.json"

DEFAULT_BALANCE_USD = 1000.0

CRYPTO_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT"
]

RRG_SYMBOLS = [
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
    "DOGE-USD"
]

RRG_BENCHMARK = "BTC-USD"


# ============================================================
# SESSION STATE
# ============================================================

saved_data = {}

if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
    except Exception:
        saved_data = {}

if "balance" not in st.session_state:
    st.session_state.balance = float(
        saved_data.get("balance", DEFAULT_BALANCE_USD)
    )

if "positions" not in st.session_state:
    st.session_state.positions = saved_data.get("positions", [])

if "trade_history" not in st.session_state:
    st.session_state.trade_history = saved_data.get("history", [])

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

try:
    if genai is not None:
        api_key = (
            st.secrets.get("GEMINI_API_KEY", "")
            or os.getenv("GEMINI_API_KEY", "")
        )

        if api_key:
            client = genai.Client(api_key=api_key)

except Exception:
    client = None


# ============================================================
# BASIC HELPERS
# ============================================================

def symbol_to_yahoo(symbol):
    return symbol.replace("USDT", "-USD")


def money(value):
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def safe_float(value, default=None):
    try:
        num = float(value)

        if np.isnan(num) or np.isinf(num):
            return default

        return num

    except Exception:
        return default


def clean_numeric_column(df, column):
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# NORMALIZE YFINANCE DATA
# ============================================================

def normalize_yfinance_data(data):
    """
    yfinance kabhi normal columns aur kabhi MultiIndex columns deta hai.
    Ye function dono ko normalize karta hai.
    """

    if data is None or data.empty:
        return None

    data = data.copy()

    try:
        # MultiIndex handling
        if isinstance(data.columns, pd.MultiIndex):

            # Example:
            # ('Close', 'BTC-USD')
            # ('Open', 'BTC-USD')

            level0 = data.columns.get_level_values(0)

            if "Close" in level0:
                data.columns = level0

            else:
                # Reverse MultiIndex possibility
                level1 = data.columns.get_level_values(1)

                if "Close" in level1:
                    data.columns = level1

        # Remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for col in required:
            if col not in data.columns:
                return None

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:
            if col in data.columns:
                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce"
                )

        data = data.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        if data.empty:
            return None

        return data

    except Exception:
        return None


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (100 / (1 + rs))

    # Extreme cases
    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False
)
def get_crypto_data(
    symbol,
    period="3mo",
    interval="1d"
):

    ticker = symbol_to_yahoo(symbol)

    try:

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        data = normalize_yfinance_data(data)

        if data is None or data.empty:
            return None

        close = data["Close"]

        # SMA
        data["SMA20"] = close.rolling(
            20,
            min_periods=20
        ).mean()

        data["SMA50"] = close.rolling(
            50,
            min_periods=50
        ).mean()

        # RSI
        data["RSI"] = calculate_rsi(
            close,
            14
        )

        # MACD
        ema12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

        data["MACD"] = ema12 - ema26

        data["MACD_Signal"] = data[
            "MACD"
        ].ewm(
            span=9,
            adjust=False
        ).mean()

        data["MACD_Hist"] = (
            data["MACD"]
            - data["MACD_Signal"]
        )

        return data

    except Exception:
        return None


# ============================================================
# LIVE PRICES
# ============================================================

@st.cache_data(
    ttl=15,
    show_spinner=False
)
def fetch_live_prices(symbols_tuple):

    tickers = [
        symbol_to_yahoo(s)
        for s in symbols_tuple
    ]

    prices = {
        s: None
        for s in symbols_tuple
    }

    try:

        data = yf.download(
            tickers=tickers,
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker",
            threads=True,
            auto_adjust=False
        )

        if data is None or data.empty:
            return prices

        # Single ticker
        if len(tickers) == 1:

            try:
                close = data["Close"].dropna()

                if not close.empty:
                    prices[symbols_tuple[0]] = float(
                        close.iloc[-1]
                    )

            except Exception:
                pass

            return prices

        # Multiple tickers
        for symbol, ticker in zip(
            symbols_tuple,
            tickers
        ):

            try:

                if ticker not in data.columns.get_level_values(0):
                    continue

                close = data[
                    ticker
                ]["Close"].dropna()

                if not close.empty:
                    prices[symbol] = float(
                        close.iloc[-1]
                    )

            except Exception:
                continue

    except Exception:
        pass

    return prices


def get_current_price(symbol):

    try:

        prices = fetch_live_prices(
            tuple(CRYPTO_SYMBOLS)
        )

        return prices.get(symbol)

    except Exception:
        return None


# ============================================================
# SWING POINTS
# ============================================================

def detect_swing_points(
    df,
    window=5
):

    df = df.copy()

    df["Swing_High"] = False
    df["Swing_Low"] = False

    if len(df) < (window * 2 + 1):
        return df

    for i in range(
        window,
        len(df) - window
    ):

        current_high = safe_float(
            df["High"].iloc[i]
        )

        current_low = safe_float(
            df["Low"].iloc[i]
        )

        if current_high is None or current_low is None:
            continue

        left_high = df[
            "High"
        ].iloc[
            i - window:i
        ].max()

        right_high = df[
            "High"
        ].iloc[
            i + 1:i + window + 1
        ].max()

        left_low = df[
            "Low"
        ].iloc[
            i - window:i
        ].min()

        right_low = df[
            "Low"
        ].iloc[
            i + 1:i + window + 1
        ].min()

        if (
            current_high > left_high
            and current_high > right_high
        ):
            df.loc[
                df.index[i],
                "Swing_High"
            ] = True

        if (
            current_low < left_low
            and current_low < right_low
        ):
            df.loc[
                df.index[i],
                "Swing_Low"
            ] = True

    return df


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_market_structure(df):

    if df is None or len(df) < 20:
        return "NEUTRAL", None, None

    df = detect_swing_points(
        df,
        window=5
    )

    swing_highs = df[
        df["Swing_High"]
    ][["High"]]

    swing_lows = df[
        df["Swing_Low"]
    ][["Low"]]

    trend = "NEUTRAL"

    # HH / HL
    if (
        len(swing_highs) >= 2
        and len(swing_lows) >= 2
    ):

        h1 = safe_float(
            swing_highs["High"].iloc[-2]
        )

        h2 = safe_float(
            swing_highs["High"].iloc[-1]
        )

        l1 = safe_float(
            swing_lows["Low"].iloc[-2]
        )

        l2 = safe_float(
            swing_lows["Low"].iloc[-1]
        )

        if (
            h2 > h1
            and l2 > l1
        ):
            trend = "BULLISH"

        elif (
            h2 < h1
            and l2 < l1
        ):
            trend = "BEARISH"

    current_price = safe_float(
        df["Close"].iloc[-1]
    )

    last_swing_high = None
    last_swing_low = None

    if len(swing_highs) > 0:
        last_swing_high = safe_float(
            swing_highs["High"].iloc[-1]
        )

    if len(swing_lows) > 0:
        last_swing_low = safe_float(
            swing_lows["Low"].iloc[-1]
        )

    bos = None
    choch = None

    if (
        trend == "BULLISH"
        and last_swing_high is not None
        and current_price is not None
        and current_price > last_swing_high
    ):
        bos = "BOS_UP"

    elif (
        trend == "BEARISH"
        and last_swing_low is not None
        and current_price is not None
        and current_price < last_swing_low
    ):
        bos = "BOS_DOWN"

    if (
        trend == "BULLISH"
        and last_swing_low is not None
        and current_price is not None
        and current_price < last_swing_low
    ):
        choch = "ChoCh_BEARISH"

    elif (
        trend == "BEARISH"
        and last_swing_high is not None
        and current_price is not None
        and current_price > last_swing_high
    ):
        choch = "ChoCh_BULLISH"

    return (
        trend,
        bos,
        choch
    )


# ============================================================
# LIQUIDITY SWEEPS
# ============================================================

def detect_liquidity_sweeps(df):

    if df is None or len(df) < 15:
        return []

    df = detect_swing_points(
        df,
        window=5
    )

    sweeps = []

    for i in range(
        1,
        len(df)
    ):

        current_high = safe_float(
            df["High"].iloc[i]
        )

        current_low = safe_float(
            df["Low"].iloc[i]
        )

        current_close = safe_float(
            df["Close"].iloc[i]
        )

        if (
            df["Swing_High"].iloc[i - 1]
            and current_high is not None
            and current_close is not None
        ):

            previous_high = safe_float(
                df["High"].iloc[i - 1]
            )

            if (
                previous_high is not None
                and current_high > previous_high
                and current_close < previous_high
            ):
                sweeps.append(
                    (
                        "BEARISH_SWEEP",
                        previous_high,
                        df.index[i]
                    )
                )

        if (
            df["Swing_Low"].iloc[i - 1]
            and current_low is not None
            and current_close is not None
        ):

            previous_low = safe_float(
                df["Low"].iloc[i - 1]
            )

            if (
                previous_low is not None
                and current_low < previous_low
                and current_close > previous_low
            ):
                sweeps.append(
                    (
                        "BULLISH_SWEEP",
                        previous_low,
                        df.index[i]
                    )
                )

    return sweeps[-3:]


# ============================================================
# FVG
# ============================================================

def detect_fvg(df):

    if df is None or len(df) < 15:
        return []

    fvg_list = []

    for i in range(
        2,
        len(df)
    ):

        first_high = safe_float(
            df["High"].iloc[i - 2]
        )

        first_low = safe_float(
            df["Low"].iloc[i - 2]
        )

        middle_open = safe_float(
            df["Open"].iloc[i - 1]
        )

        middle_close = safe_float(
            df["Close"].iloc[i - 1]
        )

        third_low = safe_float(
            df["Low"].iloc[i]
        )

        third_high = safe_float(
            df["High"].iloc[i]
        )

        if None in (
            first_high,
            first_low,
            middle_open,
            middle_close,
            third_low,
            third_high
        ):
            continue

        previous_bodies = (
            df["Close"].iloc[
                max(0, i - 10):i - 1
            ]
            -
            df["Open"].iloc[
                max(0, i - 10):i - 1
            ]
        ).abs()

        avg_body = safe_float(
            previous_bodies.mean(),
            0
        )

        middle_body = abs(
            middle_close
            - middle_open
        )

        # Avoid tiny fake FVG
        if avg_body <= 0:
            continue

        if (
            third_low > first_high
            and middle_body > avg_body * 1.5
        ):

            fvg_list.append(
                (
                    "BULLISH_FVG",
                    first_high,
                    third_low,
                    i
                )
            )

        elif (
            third_high < first_low
            and middle_body > avg_body * 1.5
        ):

            fvg_list.append(
                (
                    "BEARISH_FVG",
                    first_low,
                    third_high,
                    i
                )
            )

    return fvg_list[-3:]


# ============================================================
# ORDER BLOCK
# ============================================================

def detect_order_blocks(df):

    if df is None or len(df) < 10:
        return []

    ob_list = []

    for i in range(
        2,
        len(df)
    ):

        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_open = safe_float(
            prev["Open"]
        )

        prev_close = safe_float(
            prev["Close"]
        )

        prev_high = safe_float(
            prev["High"]
        )

        prev_low = safe_float(
            prev["Low"]
        )

        curr_open = safe_float(
            curr["Open"]
        )

        curr_close = safe_float(
            curr["Close"]
        )

        if None in (
            prev_open,
            prev_close,
            prev_high,
            prev_low,
            curr_open,
            curr_close
        ):
            continue

        # Bullish OB
        if (
            prev_close < prev_open
            and curr_close > curr_open
            and curr_close > prev_high
        ):

            ob_list.append(
                (
                    "BULLISH_OB",
                    prev_low,
                    prev_high,
                    i
                )
            )

        # Bearish OB
        elif (
            prev_close > prev_open
            and curr_close < curr_open
            and curr_close < prev_low
        ):

            ob_list.append(
                (
                    "BEARISH_OB",
                    prev_low,
                    prev_high,
                    i
                )
            )

    return ob_list[-3:]


# ============================================================
# RSI DIVERGENCE
# ============================================================

def detect_divergence(df):

    if (
        df is None
        or len(df) < 40
        or "RSI" not in df.columns
    ):
        return "NONE"

    temp = detect_swing_points(
        df,
        window=3
    )

    # Price swing lows
    lows = temp[
        temp["Swing_Low"]
    ][["Low", "RSI"]].dropna()

    # Price swing highs
    highs = temp[
        temp["Swing_High"]
    ][["High", "RSI"]].dropna()

    # Bullish divergence
    if len(lows) >= 2:

        p1 = safe_float(
            lows["Low"].iloc[-2]
        )

        p2 = safe_float(
            lows["Low"].iloc[-1]
        )

        r1 = safe_float(
            lows["RSI"].iloc[-2]
        )

        r2 = safe_float(
            lows["RSI"].iloc[-1]
        )

        if (
            None not in (p1, p2, r1, r2)
            and p2 < p1
            and r2 > r1
        ):
            return "BULLISH_DIV"

    # Bearish divergence
    if len(highs) >= 2:

        p1 = safe_float(
            highs["High"].iloc[-2]
        )

        p2 = safe_float(
            highs["High"].iloc[-1]
        )

        r1 = safe_float(
            highs["RSI"].iloc[-2]
        )

        r2 = safe_float(
            highs["RSI"].iloc[-1]
        )

        if (
            None not in (p1, p2, r1, r2)
            and p2 > p1
            and r2 < r1
        ):
            return "BEARISH_DIV"

    return "NONE"


# ============================================================
# VOLUME
# ============================================================

def analyze_volume(df):

    if (
       