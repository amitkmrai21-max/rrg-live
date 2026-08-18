import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Optional TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    talib = None
    TALIB_AVAILABLE = False

# Optional Gemini
try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Trade AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
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
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "trade_data.json"

DEFAULT_BALANCE = 100000.0

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
        saved_data.get("balance", DEFAULT_BALANCE)
    )

if "positions" not in st.session_state:
    st.session_state.positions = saved_data.get("positions", [])

if "trade_history" not in st.session_state:
    st.session_state.trade_history = saved_data.get("history", [])

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None


# ============================================================
# GEMINI
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
# HELPERS
# ============================================================

def symbol_to_yahoo(symbol):
    return symbol.replace("USDT", "-USD")


def safe_float(value, default=None):
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except Exception:
        return default


def money(value):
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


# ============================================================
# YFINANCE NORMALIZATION
# ============================================================

def normalize_yfinance_data(data):
    if data is None or data.empty:
        return None

    try:
        data = data.copy()

        if isinstance(data.columns, pd.MultiIndex):

            level0 = list(data.columns.get_level_values(0))

            if "Open" in level0:
                data.columns = data.columns.get_level_values(0)

            else:
                level1 = list(data.columns.get_level_values(1))

                if "Open" in level1:
                    data.columns = data.columns.get_level_values(1)

        data = data.loc[:, ~data.columns.duplicated()]

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for column in required:
            if column not in data.columns:
                return None

        for column in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(
                    data[column],
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

    delta = close