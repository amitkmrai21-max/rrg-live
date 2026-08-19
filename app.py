import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai
import talib

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
            "balance": float(data.get("balance", DEFAULT_BALANCE)),
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
    api_key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
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
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna(subset=["Close"])

        close = data["Close"]
        data["SMA20"] = close.rolling(20, min_periods=20).mean()
        data["SMA50"] = close.rolling(50, min_periods=50).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        data["RSI"] = 100 - (100 / (1 + rs))

        return data
    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_live_prices(symbols_tuple):
    tickers = [symbol_to_yahoo(symbol) for symbol in symbols_tuple]
    prices = {symbol: None for symbol in symbols_tuple}

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


def generate_ai_signal(pair, timeframe, price, rsi, sma20, sma50):
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
            response = clean_ai_response(result.text)
            if response:
                return response
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


def extract_close(data, ticker):
    if data is None:
        return pd.Series(dtype=float)
    try:
        return data[ticker]["Close"].dropna()
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


def build_rrg_figure(data, symbols, benchmark, day_duration_ms):
    benchmark_close = extract_close(data, benchmark)

    if benchmark_close.empty:
        return None, None, "Benchmark data available nahi hai."

    if len(benchmark_close) < 30:
        return None, None, "Benchmark ke liye sufficient data nahi hai."

    series_map = {}

    for ticker in symbols:
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
        series_map[name] = combined

    if not series_map:
        return None, None, "Coins ka RRG data calculate nahi hua."

    common_index = None
    for frame in series_map.values():
        if common_index is None:
            common_index = frame.index
        else:
            common_index = common_index.intersection(frame.index)

    if common_index is None or len(common_index) < 2:
        return None, None, "Common dates available nahi hain."

    common_index = common_index.sort_values()
    animate_days = 20
    trail = 3
    frame_dates = common_index[-animate_days:] if len(common_index) > animate_days else common_index

    if len(frame_dates) < 2:
        return None, None, "Animation ke liye sufficient dates nahi hain."

    colors = ["#00CC96", "#EF553B", "#636EFA", "#FFA15A", "#AB63FA", "#19D3F3"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(series_map.keys())}

    first_date = frame_dates.min()
    all_x = pd.concat([frame.loc[first_date:, "x"] for frame in series_map.values()])
    all_y = pd.concat([frame.loc[first_date:, "y"] for frame in series_map.values()])

    x_min = min(safe_float(all_x.min(), 98) - 1.5, 98)
    x_max = max(safe_float(all_x.max(), 102) + 1.5, 102)
    y_min = min(safe_float(all_y.min(), 98) - 1.5, 98)
    y_max = max(safe_float(all_y.max(), 102) + 1.5, 102)

    def frame_traces(date_value):
        traces = []
        for name, frame in series_map.items():
            subset = frame.loc[:date_value].tail(trail)
            if subset.empty:
                continue

            color = color_map[name]
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

    def interpolation_traces(previous_date, next_date, progress):
        traces = []
        for name, frame in series_map.items():
            if previous_date not in frame.index or next_date not in frame.index:
                continue

            fixed_trail = frame.loc[:previous_date].tail(trail - 1)

            previous_x = float(frame.loc[previous_date, "x"])
            previous_y = float(frame.loc[previous_date, "y"])
            next_x = float(frame.loc[next_date, "x"])
            next_y = float(frame.loc[next_date, "y"])

            current_x = previous_x + ((next_x - previous_x) * progress)
            current_y = previous_y + ((next_y - previous_y) * progress)

            xs = list(fixed_trail["x"])
            ys = list(fixed_trail["y"])
            xs.append(current_x)
            ys.append(current_y)

            color = color_map[name]

            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6, color=color),
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
                    marker=dict(size=13, color=color, line=dict(color="white", width=2)),
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return traces

    substeps = 4
    frame_ms = max(int(day_duration_ms / substeps), 100)

    frames = []
    slider_steps = []

    for index, current_date in enumerate(frame_dates):
        frame_name = f"d{index:03d}"
        frames.append(go.Frame(data=frame_traces(current_date), name=frame_name))

        slider_steps.append(
            {
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    },
                ],
                "label": current_date.strftime("%d/%m"),
            }
        )

        if index < len(frame_dates) - 1:
            next_date = frame_dates[index + 1]
            for substep in range(1, substeps):
                progress = substep / substeps
                sub_name = f"d{index:03d}_s{substep}"
                frames.append(
                    go.Frame(
                        data=interpolation_traces(current_date, next_date, progress),
                        name=sub_name,
                    )
                )

    figure = go.Figure(data=frame_traces(frame_dates[-1]), frames=frames)

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
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_ms, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": len(slider_steps) - 1,
                "currentvalue": {"prefix": "Date: "},
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
    )

    return figure, frame_ms, None


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Chart Analyzer",
    "🔄 Daily RRG",
    "📝 Paper Trading",
    "🤖 AI Signals",
])

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
        fig_rrg, frame_ms, error = build_rrg_figure(rrg_data, RRG_SYMBOLS, RRG_BENCHMARK, 1000)

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
            data_1d = get_crypto_data(signal_pair, period="3mo", interval="1d")
            data_4h = get_crypto_data(signal_pair, period="1mo", interval="1h")

            if data_1d is not None and not data_1d.empty:
                price_sig = get_current_price(signal_pair) or data_1d["Close"].iloc[-1]
                rsi_sig = data_1d["RSI"].iloc[-1] if "RSI" in data_1d.columns and pd.notna(data_1d["RSI"].iloc[-1]) else 50.0
                sma20_sig = data_1d["SMA20"].iloc[-1] if "SMA20" in data_1d.columns and pd.notna(data_1d["SMA20"].iloc[-1]) else price_sig
                sma50_sig = data_1d["SMA50"].iloc[-1] if "SMA50" in data_1d.columns and pd.notna(data_1d["SMA50"].iloc[-1]) else price_sig

                trend = "NEUTRAL"
                bos = None
                choch = None

                sweeps = []
                fvg = []
                ob = []
                divergence = "NONE"
                vol_spike = "NEUTRAL"
                vol_ratio = 1.0
                candle_signal = "NEUTRAL"
                confluence = "NO_DATA"
                confluence_score = 0

                ai_signal = generate_ai_signal(
                    signal_pair,
                    "1D",
                    price_sig,
                    rsi_sig,
                    sma20_sig,
                    sma50_sig,
                )

                if not ai_signal:
                    ai_signal = generate_fallback_signal(
                        price_sig,
                        rsi_sig,
                        sma20_sig,
                        sma50_sig,
                    )

                st.session_state.last_signal = ai_signal
            else:
                st.error("No data for signal generation.")

    if st.session_state.last_signal:
        st.markdown(st.session_state.last_signal)

if __name__ == "__main__":
    pass