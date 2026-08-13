import io
import json
import os
import time
from datetime import datetime
from google import genai
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# Streamlit Page Setup
st.set_page_config(
    page_title="Smart Trade AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS TO HIDE STREAMLIT BRANDING ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stAppHeader {display: none;}
            .stDeployButton {display:none;}
            div[data-testid="stDecoration"] {display: none;}
            div[data-testid="stStatusWidget"] {visibility: hidden;}
            #stDecoration {display: none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# File Persistence for Paper Trading (Prevents Data Loss on Refresh)
DATA_FILE = "trade_data.json"


def load_trade_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"balance": 10000.0, "positions": []}


def save_trade_data():
    data = {
        "balance": st.session_state.balance,
        "positions": st.session_state.positions,
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# Initialize Permanent Session State
saved_data = load_trade_data()
if "balance" not in st.session_state:
    st.session_state.balance = saved_data.get("balance", 10000.0)
if "positions" not in st.session_state:
    st.session_state.positions = saved_data.get("positions", [])

# Initialize GenAI Client using new SDK
client = None
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"⚠️ API Initialization Error: {e}")

st.title("🚀 Smart Trade AI: Assistant & Paper Trading")

crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def get_crypto_data(symbol, period="30d", interval="1d", retries=2):
    ticker_sym = symbol.replace("USDT", "-USD")
    for attempt in range(retries + 1):
        try:
            df = yf.download(
                ticker_sym, period=period, interval=interval, progress=False
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["SMA20"] = df["Close"].rolling(20).mean()
                df["SMA50"] = df["Close"].rolling(50).mean()

                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                return df
        except Exception:
            time.sleep(1)
            continue
    return None


def get_current_price(symbol, retries=2):
    ticker_sym = symbol.replace("USDT", "-USD")
    for attempt in range(retries + 1):
        try:
            ticker = yf.Ticker(ticker_sym)
            todays_data = ticker.history(period="1d")
            if not todays_data.empty:
                return float(todays_data["Close"].iloc[-1])
        except Exception:
            time.sleep(1)
            continue
    return None


tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Real-Time AI Signals",
    "💼 Live Paper Trading",
    "📊 HD Daily RRG Chart",
    "📸 Chart Analyzer",
])

# --- TAB 1: REAL-TIME AI SIGNALS ---
with tab1:
    st.subheader("💡 AI Live Buy / Sell / Target Signals")

    col1, col2 = st.columns(2)
    with col1:
        pair = st.selectbox("Crypto Pair:", crypto_symbols, key="signal_pair")
    with col2:
        timeframe = st.selectbox(
            "Timeframe:", ["15m (Scalping)", "1h (Intraday)", "1d (Swing)"]
        )

    tf_value = (
        "15m" if "15m" in timeframe else ("1h" if "1h" in timeframe else "1d")
    )
    period_value = "5d" if tf_value != "1d" else "60d"

    if st.button("🤖 AI Signal & Target Fetch Karein", key="fetch_signal"):
        with st.spinner("Market Data & AI Analysis chal raha hai..."):
            df = get_crypto_data(pair, period=period_value, interval=tf_value)
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                rsi = float(df["RSI"].iloc[-1])
                sma20 = float(df["SMA20"].iloc[-1]) if "SMA20" in df else price
                sma50 = float(df["SMA50"].iloc[-1]) if "SMA50" in df else price

                st.write("### Live Indicators:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${price:.2f}")
                c2.metric("RSI (14)", f"{rsi:.1f}")
                c3.metric(
                    "Trend", "Bullish 🟢" if sma20 > sma50 else "Bearish 🔴"
                )

                response_text = None
                if client:
                    prompt = f"""
                    Aap ek expert crypto analyst hain. Niche diye gaye real-time data par analysis karein:
                    - Pair: {pair}
                    - Timeframe: {tf_value}
                    - Current Price: ${price:.2f}
                    - RSI: {rsi:.1f}
                    - SMA20: ${sma20:.2f} | SMA50: ${sma50:.2f}
                    
                    MUST PROVIDE RESPONSE STRICTLY IN THIS FORMAT:
                    
                    ### 🟢 ACTION: [BUY / SELL / WAIT]
                    - **Entry Price Range**: [Specific Price, e.g. ${price:.2f}]
                    - **Target Price (TP)**: [Target Value, e.g. ${price*1.015:.2f}]
                    - **Stop-Loss (SL)**: [Safety Level, e.g. ${price*0.99:.2f}]
                    - **Reasoning**: [Short reason]
                    """
                    for m in [
                        "gemini-1.5-flash",
                        "gemini-2.0-flash",
                        "gemini-1.5-pro",
                    ]:
                        try:
                            res = client.models.generate_content(
                                model=m, contents=prompt
                            )
                            response_text = res.text
                            break
                        except Exception:
                            continue

                st.markdown("---")
                st.subheader("🎯 Trade Execution Levels:")

                if response_text:
                    st.success(response_text)
                else:
                    if sma20 > sma50 and rsi < 68:
                        act = "BUY 🟢"
                        entry = f"${price:.2f}"
                        tp = f"${price * 1.02:.2f}"
                        sl = f"${price * 0.99:.2f}"
                        reason = "SMA 20/50 Bullish Crossover and RSI in safe buying zone."
                    elif rsi > 70 or sma20 < sma50:
                        act = "SELL 🔴"
                        entry = f"${price:.2f}"
                        tp = f"${price * 0.98:.2f}"
                        sl = f"${price * 1.01:.2f}"
                        reason = "Overbought RSI condition / Bearish Trend momentum."
                    else:
                        act = "WAIT ⚪"
                        entry = f"${price:.2f}"
                        tp = "N/A"
                        sl = "N/A"
                        reason = "Market consolidating. Wait for breakout."

                    st.info(
                        f"### ACTION: {act}\n\n📌 **Entry**: {entry}\n\n🎯 **Target**: {tp}\n\n🛑 **Stop-Loss**: {sl}\n\n💡 **Reason**: {reason}"
                    )
            else:
                st.error("Data fetch nahi ho pa raha hai.")

# --- TAB 2: LIVE PAPER TRADING ---
with tab2:
    st.subheader("💼 Live Paper Trading Simulator")

    # Native JS Auto Refresh HTML Component (Every 5 Secs)
    if len(st.session_state.positions) > 0:
        components.html(
            """
            <script>
                setTimeout(function(){
                    window.parent.postMessage({type: 'streamlit:render'}, '*');
                }, 5000);
            </script>
            """,
            height=0,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Virtual Balance", f"${st.session_state.balance:,.2f}")

    current_pnl = 0.0
    for pos in st.session_state.positions:
        cp = get_current_price(pos["symbol"])
        if cp:
            pnl = (
                (cp - pos["entry_price"]) * pos["amount"]
                if pos["type"] == "BUY"
                else (pos["entry_price"] - cp) * pos["amount"]
            )
            current_pnl += pnl
            pos["live_pnl"] = pnl

    with c2:
        pnl_color = "#1E90FF" if current_pnl >= 0 else "#FF4500"
        pnl_prefix = "+" if current_pnl > 0 else ""
        st.markdown(
            f"#### Live Portfolio P&L: <span style='color:{pnl_color}; font-weight:bold;'>{pnl_prefix}${current_pnl:,.2f}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.write("#### ➕ Place New Trade")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        trade_pair = st.selectbox(
            "Select Crypto:", crypto_symbols, key="trade_pair"
        )
    with col2:
        trade_type = st.selectbox("Type:", ["BUY", "SELL"], key="trade_type")

    live_entry_price = get_current_price(trade_pair)
    with col3:
        st.write(
            f"**Current Price:**\n ${live_entry_price:,.2f}"
            if live_entry_price
            else "**Current Price:**\n Loading..."
        )

    with col4:
        amount_usd = st.number_input(
            "Amount (USD):", min_value=10.0, step=10.0, value=100.0
        )

    if st.button(f"Place {trade_type} Order", key="place_order_btn"):
        if not live_entry_price:
            st.warning("Live Price load hone ka wait karein.")
        elif amount_usd > st.session_state.balance:
            st.error("Insufficient balance!")
        else:
            st.session_state.balance -= amount_usd
            st.session_state.positions.append({
                "id": len(st.session_state.positions) + 1,
                "symbol": trade_pair,
                "type": trade_type,
                "amount_usd": amount_usd,
                "amount": amount_usd / live_entry_price,
                "entry_price": live_entry_price,
                "time": datetime.now().strftime("%H:%M:%S"),
                "live_pnl": 0.0,
            })
            save_trade_data()
            st.success("Trade placed successfully!")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.write("#### 📂 Open Positions")
    if not st.session_state.positions:
        st.info("No open positions.")
    else:
        for p in st.session_state.positions:
            pnl_val = p.get("live_pnl", 0.0)
            color = "#1E90FF" if pnl_val >= 0 else "#FF4500"
            prefix = "+" if pnl_val > 0 else ""

            col_a, col_b, col_c, col_d, col_e, col_f = st.columns(
                [1, 2, 1, 2, 2, 2]
            )
            col_a.write(f"**#{p['id']}**")
            col_b.write(f"**{p['symbol']}**")
            col_c.write(f"**{p['type']}**")
            col_d.write(f"${p['amount_usd']:,.2f}")
            col_e.write(f"${p['entry_price']:,.2f}")
            col_f.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{prefix}${pnl_val:,.2f}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_c1, _ = st.columns([3, 1])
        with col_c1:
            pos_id = st.selectbox(
                "Select Position ID to close:",
                [p["id"] for p in st.session_state.positions],
            )

        if st.button("Close Selected Position", key="close_pos_btn"):
            for i, p in enumerate(st.session_state.positions):
                if p["id"] == pos_id:
                    cp = get_current_price(p["symbol"])
                    if cp:
                        pnl = (
                            (cp - p["entry_price"]) * p["amount"]
                            if p["type"] == "BUY"
                            else (p["entry_price"] - cp) * p["amount"]
                        )
                        st.session_state.balance += p["amount_usd"] + pnl
                        st.session_state.positions.pop(i)
                        save_trade_data()
                        st.success("Position closed!")
                        time.sleep(1)
                        st.rerun()
                    break

# --- TAB 3: HD RRG CHART ---
with tab3:
    st.subheader("📊 Ultra-HD Daily RRG Rotation Chart")
    if st.button("Generate Large HD RRG Chart", key="gen_rrg_btn"):
        with st.spinner("Plotting chart..."):
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
            benchmark = "USDT-USD"
            try:
                bm_data = yf.download(benchmark, period="60d", progress=False)
                if isinstance(bm_data.columns, pd.MultiIndex):
                    bm_data.columns = bm_data.columns.get_level_values(0)
                bm_close = bm_data["Close"]

                fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
                ax.axhline(100, color="gray", linestyle="--")
                ax.axvline(100, color="gray", linestyle="--")

                ax.text(
                    101.5,
                    101.5,
                    "LEADING 🟢",
                    color="green",
                    fontsize=12,
                    weight="bold",
                )
                ax.text(
                    97.5,
                    101.5,
                    "WEAKENING 🟠",
                    color="orange",
                    fontsize=12,
                    weight="bold",
                )
                ax.text(
                    97.5,
                    98.5,
                    "LAGGING 🔴",
                    color="red",
                    fontsize=12,
                    weight="bold",
                )
                ax.text(
                    101.5,
                    98.5,
                    "IMPROVING 🔵",
                    color="blue",
                    fontsize=12,
                    weight="bold",
                )

                for sym in symbols:
                    s_data = yf.download(sym, period="60d", progress=False)
                    if isinstance(s_data.columns, pd.MultiIndex):
                        s_data.columns = s_data.columns.get_level_values(0)
                    s_close = s_data["Close"]

                    rs = (s_close / bm_close) * 100
                    rs_ratio = (rs / rs.rolling(14).mean()) * 100
                    rs_momentum = (rs_ratio / rs_ratio.shift(1)) * 100

                    x_vals = rs_ratio.iloc[-7:].values
                    y_vals = rs_momentum.iloc[-7:].values

                    coin_name = sym.replace("-USD", "")

                    ax.plot(
                        x_vals,
                        y_vals,
                        label=coin_name,
                        linewidth=2.5,
                        marker="o",
                        markersize=4,
                    )

                    ax.annotate(
                        coin_name,
                        (x_vals[-1], y_vals[-1]),
                        xytext=(x_vals[-1] + 0.15, y_vals[-1] + 0.15),
                        fontsize=11,
                        weight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            fc="yellow",
                            ec="black",
                            lw=1,
                            alpha=0.8,
                        ),
                    )

                ax.set_xlabel("JDK RS-Ratio", fontsize=11, weight="bold")
                ax.set_ylabel("JDK RS-Momentum", fontsize=11, weight="bold")
                ax.legend(loc="upper left")
                ax.grid(True, linestyle=":", alpha=0.5)

                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 4: CHART ANALYZER ---
with tab4:
    st.subheader("📸 Screenshot Analyzer")
    uploaded_file = st.file_uploader(
        "Upload Chart Image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file and client:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

        if st.button("Analyze Image with AI", key="analyze_img"):
            with st.spinner("Analyzing image..."):
                prompt = "Aap ek professional trading analyst hain. Is chart ko analyze karke BUY/SELL/WAIT decision, Entry, Target, aur Stop loss bataayein."
                for m in ["gemini-1.5-flash", "gemini-2.5-flash-light"]:
                    try:
                        resp = client.models.generate_content(
                            model=m, contents=[img, prompt]
                        )
                        st.success("Analysis Result:")
                        st.markdown(resp.text)
                        break
                    except Exception as err:
                        st.error(f"Error: {err}")
