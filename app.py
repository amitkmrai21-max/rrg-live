import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from PIL import Image
from google import genai


st.set_page_config(
    page_title="Smart Trade AI",
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
        return {
            "balance": DEFAULT_BALANCE,
            "positions": [],
            "history": [],
        }

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "balance": float(
                data.get("balance", DEFAULT_BALANCE)
            ),
            "positions": data.get("positions", []),
            "history": data.get("history", []),
        }

    except Exception:
        return {
            "balance": DEFAULT_BALANCE,
            "positions": [],
            "history": [],
        }


def save_trade_data():
    try:
        data = {
            "balance": st.session_state.balance,
            "positions": st.session_state.positions,
            "history": st.session_state.trade_history,
        }

        with open(DATA_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

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

if "last_chart_analysis" not in st.session_state:
    st.session_state.last_chart_analysis = None

if "rrg_summary" not in st.session_state:
    st.session_state.rrg_summary = None

if "rrg_generated_at" not in st.session_state:
    st.session_state.rrg_generated_at = None


client = None

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = os.getenv("GEMINI_API_KEY", "")

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        client = None


def symbol_to_yahoo(symbol):
    return symbol.replace("USDT", "-USD")


def clean_ai_response(text):
    if not text:
        return None

    text = str(text).strip()

    if len(text) > 6000:
        text = text[:6000]


    return text


def money(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "₹0.00"


def safe_float(value, default=None):
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except Exception:
        return default


@st.cache_data(ttl=60, show_spinner=False)
def get_crypto_data(symbol, period="30d", interval="1d"):
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
        data["Close"] = pd.to_numeric(
            data["Close"],
            errors="coerce",
        )
        data = data.dropna(subset=["Close"])

        close = data["Close"]

        data["SMA20"] = close.rolling(
            20,
            min_periods=20,
        ).mean()

        data["SMA50"] = close.rolling(
            50,
            min_periods=50,
        ).mean()

        delta = close.diff()

        gain = delta.clip(
            lower=0
        ).rolling(
            14,
            min_periods=14,
        ).mean()

        loss = (-delta.clip(upper=0)).rolling(
            14,
            min_periods=14,
        ).mean()

        rs = gain / loss.replace(0, np.nan)

        data["RSI"] = 100 - (100 / (1 + rs))

        return data

    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_live_prices(symbols_tuple):
    tickers = [
        symbol_to_yahoo(symbol)
        for symbol in symbols_tuple
    ]

    prices = {
        symbol: None
        for symbol in symbols_tuple
    }

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
                    prices[symbol] = float(
                        close.iloc[-1]
                    )

            except Exception:
                continue

    except Exception:
        pass

    return prices


def get_current_price(symbol):
    prices = fetch_live_prices(
        tuple(CRYPTO_SYMBOLS)
    )
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


def generate_ai_signal(
    pair,
    timeframe,
    price,
    rsi,
    sma20,
    sma50,
):
    if not client:
        return None

    prompt = f"""
You are a professional crypto technical analyst.

Analyze this market data:

Pair: {pair}
Timeframe: {timeframe}
Current Price: ${price:.6f}
RSI: {rsi:.2f}
SMA20: ${sma20:.6f}
SMA50: ${sma50:.6f}

Give exactly one final action: BUY, SELL, or WAIT.

Do not give conditional alternatives.
Do not say both BUY and SELL.
If the data is mixed, choose WAIT.

Reply strictly in this format:

### ACTION: [BUY / SELL / WAIT]
- Entry: [price or range]
- Target: [price]
- Stop-Loss: [price]
- Confidence: [0-100]%
- Reasoning: [short explanation in Hinglish]
"""

    models = [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
    ]

    for model_name in models:
        try:
            result = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            response = clean_ai_response(
                result.text
            )

            if response:
                return response

        except Exception:
            continue

    return None


def generate_fallback_signal(
    price,
    rsi,
    sma20,
    sma50,
):
    if sma20 > sma50 and rsi < 68:
        action = "BUY 🟢"
        entry = f"${price:.6f}"
        target = f"${price * 1.02:.6f}"
        stop = f"${price * 0.99:.6f}"
        reason = (
            "SMA20 SMA50 ke upar hai aur RSI "
            "overbought zone me nahi hai."
        )

    elif sma20 < sma50 or rsi > 70:
        action = "SELL 🔴"
        entry = f"${price:.6f}"
        target = f"${price * 0.98:.6f}"
        stop = f"${price * 1.01:.6f}"
        reason = (
            "Trend weak hai ya RSI overbought "
            "zone me hai."
        )

    else:
        action = "WAIT ⚪"
        entry = f"${price:.6f}"
        target = "N/A"
        stop = "N/A"
        reason = (
            "Market mixed/consolidating hai. "
            "Confirmation ka wait karein."
        )

    return f"""
### ACTION: {action}

- **Entry**: {entry}
- **Target**: {target}
- **Stop-Loss**: {stop}
- **Reasoning**: {reason}
"""


def extract_close(data, ticker):
    if data is None:
        return pd.Series(dtype=float)

    try:
        series = data[ticker]["Close"]
        return series.dropna()

    except Exception:
        return pd.Series(dtype=float)


def quadrant(x_value, y_value):
    if x_value >= 100 and y_value >= 100:
        return "LEADING 🟢"

    if x_value < 100 and y_value >= 100:
        return "WEAKENING 🟠"

    if x_value < 100 and y_value < 100:
        return "LAGGING 🔴"

    return "IMPROVING 🔵"


def build_rrg_figure(
    data,
    symbols,
    benchmark,
    day_duration_ms,
):
    benchmark_close = extract_close(
        data,
        benchmark,
    )

    if benchmark_close.empty:
        return None, None, (
            "Benchmark data available nahi hai."
        )

    if len(benchmark_close) < 30:
        return None, None, (
            "Benchmark ke liye sufficient data nahi hai."
        )

    series_map = {}

    for ticker in symbols:
        coin_close = extract_close(
            data,
            ticker,
        )

        if coin_close.empty:
            continue

        relative_strength = (
            coin_close / benchmark_close
        ) * 100

        relative_strength = (
            relative_strength.dropna()
        )

        if len(relative_strength) < 30:
            continue

        rs_ratio = (
            relative_strength
            / relative_strength.rolling(
                14,
                min_periods=14,
            ).mean()
        ) * 100

        rs_momentum = (
            rs_ratio / rs_ratio.shift(1)
        ) * 100

        combined = pd.concat(
            [rs_ratio, rs_momentum],
            axis=1,
        ).dropna()

        if combined.empty:
            continue

        combined.columns = ["x", "y"]

        name = ticker.replace(
            "-USD",
            "",
        )

        series_map[name] = combined

    if not series_map:
        return None, None, (
            "Coins ka RRG data calculate nahi hua."
        )

    common_index = None

    for frame in series_map.values():
        if common_index is None:
            common_index = frame.index
        else:
            common_index = common_index.intersection(
                frame.index
            )

    if common_index is None:
        return None, None, (
            "Common dates available nahi hain."
        )

    common_index = common_index.sort_values()

    animate_days = 20
    trail = 3

    if len(common_index) > animate_days:
        frame_dates = common_index[-animate_days:]
    else:
        frame_dates = common_index

    if len(frame_dates) < 2:
        return None, None, (
            "Animation ke liye sufficient dates nahi hain."
        )

    colors = [
        "#00CC96",
        "#EF553B",
        "#636EFA",
        "#FFA15A",
        "#AB63FA",
        "#19D3F3",
    ]

    color_map = {}

    for index, name in enumerate(
        series_map.keys()
    ):
        color_map[name] = colors[
            index % len(colors)
        ]

    first_date = frame_dates.min()

    all_x = pd.concat(
        [
            frame.loc[first_date:, "x"]
            for frame in series_map.values()
        ]
    )

    all_y = pd.concat(
        [
            frame.loc[first_date:, "y"]
            for frame in series_map.values()
        ]
    )

    x_min = safe_float(
        all_x.min(),
        98,
    ) - 1.5

    x_max = safe_float(
        all_x.max(),
        102,
    ) + 1.5

    y_min = safe_float(
        all_y.min(),
        98,
    ) - 1.5

    y_max = safe_float(
        all_y.max(),
        102,
    ) + 1.5

    x_min = min(x_min, 98)
    x_max = max(x_max, 102)
    y_min = min(y_min, 98)
    y_max = max(y_max, 102)

    def frame_traces(date_value):
        traces = []

        for name, frame in series_map.items():
            subset = frame.loc[:date_value].tail(
                trail
            )

            if subset.empty:
                continue

            color = color_map[name]

            dates_text = [
                date.strftime("%d/%m")
                for date in subset.index
            ]

            traces.append(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="lines+markers",
                    line=dict(
                        color=color,
                        width=2.5,
                    ),
                    marker=dict(
                        size=6,
                        color=color,
                    ),
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
                    marker=dict(
                        size=13,
                        color=color,
                        line=dict(
                            color="white",
                            width=2,
                        ),
                    ),
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

    def interpolation_traces(
        previous_date,
        next_date,
        progress,
    ):
        traces = []

        for name, frame in series_map.items():
            if previous_date not in frame.index:
                continue

            if next_date not in frame.index:
                continue

            fixed_trail = frame.loc[
                :previous_date
            ].tail(trail - 1)

            previous_x = float(
                frame.loc[previous_date, "x"]
            )

            previous_y = float(
                frame.loc[previous_date, "y"]
            )

            next_x = float(
                frame.loc[next_date, "x"]
            )

            next_y = float(
                frame.loc[next_date, "y"]
            )

            current_x = previous_x + (
                (next_x - previous_x) * progress
            )

            current_y = previous_y + (
                (next_y - previous_y) * progress
            )

            xs = list(fixed_trail["x"])
            xs.append(current_x)

            ys = list(fixed_trail["y"])
            ys.append(current_y)

            color = color_map[name]

            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(
                        color=color,
                        width=2.5,
                    ),
                    marker=dict(
                        size=6,
                        color=color,
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            traces.append(
                go.Scatter(
                    x=[current_x],
                    y=[current_y],
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=color,
                        line=dict(
                            color="white",
                            width=2,
                        ),
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return traces

    substeps = 4

    frame_ms = max(
        int(day_duration_ms / substeps),
        100,
    )

    frames = []
    slider_steps = []

    for index, current_date in enumerate(
        frame_dates
    ):
        frame_name = f"d{index:03d}"

        frames.append(
            go.Frame(
                data=frame_traces(current_date),
                name=frame_name,
            )
        )

        slider_steps.append(
            {
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "frame": {
                            "duration": 0,
                            "redraw": True,
                        },
                        "mode": "immediate",
                    },
                ],
                "label": current_date.strftime(
                    "%d/%m"
                ),
            }
        )

        if index < len(frame_dates) - 1:
            next_date = frame_dates[index + 1]

            for substep in range(1, substeps):
                progress = substep / substeps
                sub_name = (
                    f"d{index:03d}_s{substep}"
                )

                frames.append(
                    go.Frame(
                        data=interpolation_traces(
                            current_date,
                            next_date,
                            progress,
                        ),
                        name=sub_name,
                    )
                )

    figure = go.Figure(
        data=frame_traces(frame_dates[-1]),
        frames=frames,
    )

    figure.add_shape(
        type="line",
        x0=100,
        x1=100,
        y0=y_min,
        y1=y_max,
        line=dict(
            color="gray",
            dash="dash",
        ),
    )

    figure.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=100,
        y1=100,
        line=dict(
            color="gray",
            dash="dash",
        ),
    )

    figure.add_shape(
        type="rect",
        x0=100,
        x1=x_max,
        y0=100,
        y1=y_max,
        fillcolor="#2ecc71",
        opacity=0.10,
        line_width=0,
        layer="below",
    )

    figure.add_shape(
        type="rect",
        x0=x_min,
        x1=100,
        y0=100,
        y1=y_max,
        fillcolor="#f39c12",
        opacity=0.10,
        line_width=0,
        layer="below",
    )

    figure.add_shape(
        type="rect",
        x0=x_min,
        x1=100,
        y0=y_min,
        y1=100,
        fillcol