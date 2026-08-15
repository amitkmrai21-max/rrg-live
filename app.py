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
        text = text[:6000] + "


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
        fillcolor="#e74c3c",
        opacity=0.10,
        line_width=0,
        layer="below",
    )

    figure.add_shape(
        type="rect",
        x0=100,
        x1=x_max,
        y0=y_min,
        y1=100,
        fillcolor="#3498db",
        opacity=0.10,
        line_width=0,
        layer="below",
    )

    figure.add_annotation(
        x=x_max,
        y=y_max,
        text="LEADING",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(
            color="#2ecc71",
            size=14,
        ),
    )

    figure.add_annotation(
        x=x_min,
        y=y_max,
        text="WEAKENING",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(
            color="#f39c12",
            size=14,
        ),
    )

    figure.add_annotation(
        x=x_min,
        y=y_min,
        text="LAGGING",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(
            color="#e74c3c",
            size=14,
        ),
    )

    figure.add_annotation(
        x=x_max,
        y=y_min,
        text="IMPROVING",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(
            color="#3498db",
            size=14,
        ),
    )

    figure.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(
            l=10,
            r=10,
            t=70,
            b=30,
        ),
        xaxis=dict(
            title="RS-Ratio",
            range=[x_min, x_max],
            fixedrange=False,
        ),
        yaxis=dict(
            title="RS-Momentum",
            range=[y_min, y_max],
            fixedrange=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": -0.14,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": frame_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 100,
                                    "easing": "linear",
                                },
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.12,
                "y": -0.07,
                "len": 0.85,
                "steps": slider_steps,
            }
        ],
    )

    latest_date = frame_dates[-1]
    summary_rows = []

    for name, frame in series_map.items():
        if latest_date not in frame.index:
            continue

        x_value = float(
            frame.loc[latest_date, "x"]
        )

        y_value = float(
            frame.loc[latest_date, "y"]
        )

        summary_rows.append(
            {
                "Coin": name,
                "Status": quadrant(
                    x_value,
                    y_value,
                ),
                "RS-Ratio": round(
                    x_value,
                    2,
                ),
                "RS-Momentum": round(
                    y_value,
                    2,
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return figure, summary, None


def calculate_position_pnl(
    position,
    current_price,
):
    if current_price is None:
        return 0.0

    quantity = float(
        position.get("amount", 0)
    )

    entry_price = float(
        position.get("entry_price", 0)
    )

    if position.get("type") == "BUY":
        return (
            current_price - entry_price
        ) * quantity

    return (
        entry_price - current_price
    ) * quantity


def render_live_paper_prices():
    prices = fetch_live_prices(
        tuple(CRYPTO_SYMBOLS)
    )

    current_pnl = 0.0

    for position in st.session_state.positions:
        symbol = position.get("symbol")
        current_price = prices.get(symbol)

        if current_price is None:
            continue

        pnl = calculate_position_pnl(
            position,
            current_price,
        )

        position["live_price"] = float(
            current_price
        )

        position["live_pnl"] = float(pnl)

        current_pnl += pnl

    color = (
        "#1E90FF"
        if current_pnl >= 0
        else "#FF4500"
    )

    prefix = (
        "+"
        if current_pnl > 0
        else ""
    )

    st.markdown(
        f"""
        <h4 style="color:{color};">
        Live Portfolio P&amp;L:
        {prefix}{money(current_pnl)}
        </h4>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Prices update approximately every 10 seconds. "
        "Last checked: "
        + datetime.now().strftime(
            "%H:%M:%S"
        )
    )


st.title(
    "🚀 Smart Trade AI: "
    "Assistant & Paper Trading"
)

st.caption(
    "AI signals, paper trading, RRG rotation "
    "aur chart analysis ek hi app me."
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🤖 Real-Time AI Signals",
        "💼 Live Paper Trading",
        "📊 HD Daily RRG Chart",
        "📸 Chart Analyzer",
    ]
)


with tab1:
    st.subheader(
        "💡 AI Live Buy / Sell / Target Signals"
    )

    col1, col2 = st.columns(2)

    with col1:
        pair = st.selectbox(
            "Crypto Pair:",
            CRYPTO_SYMBOLS,
            key="signal_pair",
        )

    with col2:
        timeframe = st.selectbox(
            "Timeframe:",
            [
                "15m (Scalping)",
                "1h (Intraday)",
                "1d (Swing)",
            ],
            key="signal_timeframe",
        )

    if st.button(
        "AI Signal & Target Fetch Karein",
        key="fetch_signal",
        use_container_width=True,
    ):
        timeframe_value = (
            "15m"
            if "15m" in timeframe
            else "1h"
            if "1h" in timeframe
            else "1d"
        )

        period_value = (
            "5d"
            if timeframe_value != "1d"
            else "90d"
        )

        with st.spinner(
            "Market data aur AI analysis chal raha hai..."
        ):
            market_data = get_crypto_data(
                pair,
                period=period_value,
                interval=timeframe_value,
            )

            if market_data is None:
                st.error(
                    "Market data nahi mil raha."
                )

            else:
                valid_data = market_data.dropna(
                    subset=[
                        "Close",
                        "RSI",
                        "SMA20",
                        "SMA50",
                    ]
                )

                if valid_data.empty:
                    st.error(
                        "Indicators ke liye sufficient "
                        "data nahi hai."
                    )

                else:
                    latest = valid_data.iloc[-1]

                    price = float(
                        latest["Close"]
                    )

                    rsi = float(
                        latest["RSI"]
                    )

                    sma20 = float(
                        latest["SMA20"]
                    )

                    sma50 = float(
                        latest["SMA50"]
                    )

                    trend = (
                        "Bullish"
                        if sma20 > sma50
                        else "Bearish"
                    )

                    ai_response = generate_ai_signal(
                        pair,
                        timeframe_value,
                        price,
                        rsi,
                        sma20,
                        sma50,
                    )

                    fallback_response = (
                        generate_fallback_signal(
                            price,
                            rsi,
                            sma20,
                            sma50,
                        )
                    )

                    st.session_state.last_signal = {
                        "pair": pair,
                        "timeframe": timeframe,
                        "price": price,
                        "rsi": rsi,
                        "sma20": sma20,
                        "sma50": sma50,
                        "trend": trend,
                        "ai_response": ai_response,
                        "fallback_response": (
                            fallback_response
                        ),
                        "updated": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                    }

    signal = st.session_state.last_signal

    if signal:
        st.caption(
            "Last updated: "
            + signal["updated"]
            + " | "
            + signal["pair"]
            + " | "
            + signal["timeframe"]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current Price",
            f"${signal['price']:.6f}",
        )

        c2.metric(
            "RSI",
            f"{signal['rsi']:.2f}",
        )

        c3.metric(
            "SMA20",
            f"${signal['sma20']:.6f}",
        )

        c4.metric(
            "Trend",
            signal["trend"],
        )

        st.markdown("---")
        st.subheader(
            "🎯 Trade Execution Levels"
        )

        if signal["ai_response"]:
            st.success(
                signal["ai_response"]
            )
        else:
            st.info(
                signal["fallback_response"]
            )

        st.caption(
            "Educational analysis only. "
            "Guaranteed financial advice nahi hai."
        )

    else:
        st.info(
            "Signal generate karne ke liye "
            "button dabayein."
        )


with tab2:
    st.subheader(
        "💼 Live Paper Trading Simulator"
    )

    top1, top2, top3 = st.columns(3)

    top1.metric(
        "Virtual Balance",
        money(st.session_state.balance),
    )

    invested_amount = 0.0

    for position in st.session_state.positions:
        invested_amount += float(
            position.get(
                "amount_inr",
                position.get(
                    "amount_usd",
                    0,
                ),
            )
        )

    top2.metric(
        "Invested Amount",
        money(invested_amount),
    )

    with top3:
        if st.button(
            "Reset Portfolio",
            key="reset_portfolio",
            use_container_width=True,
        ):
            st.session_state.balance = (
                DEFAULT_BALANCE
            )

            st.session_state.positions = []
            st.session_state.trade_history = []

            save_trade_data()

            st.success(
                "Portfolio reset ho gaya."
            )

            st.rerun()

    if st.session_state.positions:
        render_live_paper_prices()

    st.markdown("---")
    st.write("#### Place New Trade")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        trade_pair = st.selectbox(
            "Select Crypto:",
            CRYPTO_SYMBOLS,
            key="trade_pair",
        )

    with col2:
        trade_type = st.selectbox(
            "Type:",
            ["BUY", "SELL"],
            key="trade_type",
        )

    current_price = get_current_price(
        trade_pair
    )

    with col3:
        if current_price is not None:
            st.metric(
                "Current Price",
                f"${current_price:.6f}",
            )
        else:
            st.warning(
                "Price loading..."
            )

    with col4:
        trade_amount = st.number_input(
            "Trade Amount INR:",
            min_value=10.0,
            step=10.0,
            value=1000.0,
            key="trade_amount",
        )

    if st.button(
        "Place Order",
        key="place_order",
        use_container_width=True,
    ):
        if current_price is None:
            st.error(
                "Live price available nahi hai."
            )

        elif trade_amount > st.session_state.balance:
            st.error(
                "Insufficient virtual balance."
            )

        else:
            quantity = (
                trade_amount / current_price
            )

            position = {
                "id": int(time.time()),
                "symbol": trade_pair,
                "type": trade_type,
                "amount_inr": float(
                    trade_amount
                ),
                "amount": float(quantity),
                "entry_price": float(
                    current_price
                ),
                "live_price": float(
                    current_price
                ),
                "live_pnl": 0.0,
                "time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
            }

            st.session_state.balance -= float(
                trade_amount
            )

            st.session_state.positions.append(
                position
            )

            save_trade_data()

            st.success(
                "Paper trade placed successfully."
            )

            st.rerun()

    st.markdown("---")
    st.write("#### Open Positions")

    if not st.session_state.positions:
        st.info("No open positions.")

    else:
        for position in st.session_state.positions:
            amount_inr = float(
                position.get(
                    "amount_inr",
                    position.get(
                        "amount_usd",
                        0,
                    ),
                )
            )

            live_price = position.get(
                "live_price"
            )

            live_pnl = float(
                position.get(
                    "live_pnl",
                    0,
                )
            )

            color = (
                "#1E90FF"
                if live_pnl >= 0
                else "#FF4500"
            )

            prefix = (
                "+"
                if live_pnl > 0
                else ""
            )

            if live_price is not None:
                live_price_text = (
                    f"${float(live_price):.6f}"
                )
            else:
                live_price_text = (
                    "Loading..."
                )

            entry_price_text = (
                f"${float(position.get('entry_price', 0)):.6f}"
            )

            st.markdown(
                f"""
                <div style="
                    border: 1px solid #444;
                    border-radius: 10px;
                    padding: 12px;
                    margin-bottom: 10px;
                ">
                    <b>#{position.get('id')}</b>
                    &nbsp; {position.get('symbol')}
                    &nbsp; {position.get('type')}
                    <br>
                    Invested: {money(amount_inr)}
                    &nbsp; | &nbsp;
                    Quantity:
                    {float(position.get('amount', 0)):.8f}
                    <br>
                    Entry: {entry_price_text}
                    &nbsp; | &nbsp;
                    Live: {live_price_text}
                    <br>
                    <span style="color:{color}; font-weight:bold;">
                        Live P&amp;L:
                        {prefix}{money(live_pnl)}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        position_ids = [
            position.get("id")
            for position in st.session_state.positions
        ]

        selected_position_id = st.selectbox(
            "Select Position ID to close:",
            position_ids,
            key="selected_position",
        )

        if st.button(
            "Close Selected Position",
            key="close_position",
            use_container_width=True,
        ):
            for index, position in enumerate(
                st.session_state.positions
            ):
                if position.get("id") != selected_position_id:
                    continue

                close_price = get_current_price(
                    position.get("symbol")
                )

                if close_price is None:
                    st.error(
                        "Closing price available nahi hai."
                    )
                    break

                pnl = calculate_position_pnl(
                    position,
                    close_price,
                )

                invested = float(
                    position.get(
                        "amount_inr",
                        position.get(
                            "amount_usd",
                            0,
                        ),
                    )
                )

                st.session_state.balance += (
                    invested + pnl
                )

                history_item = {
                    **position,
                    "close_price": float(
                        close_price
                    ),
                    "realized_pnl": float(
                        pnl
                    ),
                    "closed_at": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                }

                st.session_state.trade_history.append(
                    history_item
                )

                st.session_state.positions.pop(
                    index
                )

                save_trade_data()

                st.success(
                    "Position closed successfully."
                )

                st.rerun()

    if st.session_state.trade_history:
        with st.expander(
            "Trade History"
        ):
            history_rows = []

            for trade in st.session_state.trade_history:
                history_rows.append(
                    {
                        "Symbol": trade.get(
                            "symbol"
                        ),
                        "Type": trade.get(
                            "type"
                        ),
                        "Entry": trade.get(
                            "entry_price"
                        ),
                        "Exit": trade.get(
                            "close_price"
                        ),
                        "P&L": trade.get(
                            "realized_pnl"
                        ),
                        "Closed At": trade.get(
                            "closed_at"
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(history_rows),
                use_container_width=True,
                hide_index=True,
            )


with tab3:
    st.subheader(
        "📊 Ultra-HD Daily RRG Rotation Chart"
    )

    st.caption(
        "Chart ko mouse/finger se zoom aur pan karein. "
        "Play button se daily movement dekhein."
    )

    speed = st.selectbox(
        "Play Speed:",
        ["Slow", "Extra Slow"],
        key="rrg_speed",
    )

    day_duration = (
        2200
        if speed == "Slow"
        else 3800
    )

    if st.button(
        "Generate / Refresh RRG Chart",
        key="generate_rrg",
        use_container_width=True,
    ):
        with st.spinner(
            "RRG chart generate ho raha hai..."
        ):
            all_tickers = RRG_SYMBOLS + [
                RRG_BENCHMARK
            ]

            rrg_data = download_rrg_data(
                tuple(all_tickers)
            )

            if rrg_data is None or rrg_data.empty:
                st.error(
                    "Yahoo Finance se RRG data nahi mila."
                )

            else:
                figure, summary, error = (
                    build_rrg_figure(
                        rrg_data,
                        RRG_SYMBOLS,
                        RRG_BENCHMARK,
                        day_duration,
                    )
                )

                if error:
                    st.error(error)

                else:
                    st.session_state.rrg_summary = (
                        summary
                    )

                    st.session_state.rrg_generated_at = (
                        datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config={
                            "scrollZoom": True,
                            "displaylogo": False,
                            "responsive": True,
                        },
                        key="rrg_chart",
                    )

    if st.session_state.rrg_summary is not None:
        st.caption(
            "Last generated: "
            + str(
                st.session_state.rrg_generated_at
            )
        )

        st.write("#### Current RRG Summary")

        st.dataframe(
            st.session_state.rrg_summary,
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "RRG benchmark BTC-USD hai. Yahoo Finance data "
        "delayed ho sakta hai."
    )


with tab4:
    st.subheader(
        "📸 Screenshot Analyzer"
    )

    uploaded_file = st.file_uploader(
        "Upload Chart Image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="chart_upload",
    )

    if uploaded_file:
        try:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

            if image.width < 300 or image.height < 200:
                st.warning(
                    "Clear aur larger chart screenshot "
                    "upload karein."
                )

            image.thumbnail(
                (1600, 1600)
            )

            st.image(
                image,
                use_container_width=True,
            )

            if client is None:
                st.warning(
                    "Gemini API key configured nahi hai."
                )

            elif st.button(
                "Analyze Image with AI",
                key="analyze_image",
                use_container_width=True,
            ):
                prompt = """
You are a professional crypto technical analyst.

Analyze the uploaded chart carefully.

Rules:
- Choose exactly one action: BUY, SELL, or WAIT.
- Never provide BUY and SELL together.
- Never provide conditional alternatives.
- If the chart is unclear or mixed, choose WAIT.
- Do not invent a price if the price scale is not visible.
- If a value is not visible, write Not visible.
- Provide output in Hinglish.
- Keep the analysis concise.

Reply strictly in this format:

### ACTION: [BUY / SELL / WAIT]

- Entry: [price or Not visible]
- Target (TP): [price or Not visible]
- Stop-Loss (SL): [price or Not visible]
- Confidence: [0-100]%
- Reasoning: [2-3 clear lines]
"""

                with st.spinner(
                    "Chart analyze ho raha hai..."
                ):
                    response_text = None

                    models = [
                        "gemini-flash-lite-latest",
                        "gemini-flash-latest",
                    ]

                    for model_name in models:
                        try:
                            result = (
                                client.models.generate_content(
                                    model=model_name,
                                    contents=[
                                        image,
                                        prompt,
                                    ],
                                )
                            )

                            response_text = (
                                clean_ai_response(
                                    result.text
                                )
                            )

                            if response_text:
                                break

                        except Exception:
                            continue

                    if response_text:
                        st.session_state.last_chart_analysis = (
                            response_text
                        )
                    else:
                        st.error(
                            "AI analysis complete nahi ho paya."
                        )

        except Exception:
            st.error(
                "Image open nahi ho paayi."
            )

    if st.session_state.last_chart_analysis:
        st.markdown("---")
        st.success("Analysis Result:")
        st.markdown(
            st.session_state.last_chart_analysis
        )

        st.caption(
            "Educational analysis hai, financial advice nahi."
        )