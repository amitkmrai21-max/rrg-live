import io
import time
from PIL import Image
from google import genai
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

# Streamlit Page Setup
st.set_page_config(
    page_title="Smart Trade",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS TO HIDE STREAMLIT BRANDING & LOGOS ---
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

# Gemini API Initialization
client = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("⚠️ GEMINI_API_KEY Streamlit Secrets me set nahi hai!")

st.title("🚀 AI Trading Assistant & RRG Rotation")

tab1, tab2, tab3 = st.tabs([
    "🤖 Real-Time AI Signals",
    "📊 HD Daily RRG Chart (Clear BTC)",
    "📸 Chart Analyzer",
])


# Helper Function: Fetch Real-Time Data via yfinance
def get_crypto_data(symbol, period="30d", interval="1d"):
    ticker = symbol.replace("USDT", "-USD")
    try:
        df = yf.download(
            ticker, period=period, interval=interval, progress=False
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Technical Indicators
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        return df
    except Exception:
        return None


# --- TAB 1: REAL-TIME AI SIGNALS ---
with tab1:
    st.subheader("💡 AI Live Buy / Sell / Target Signals")

    col1, col2 = st.columns(2)
    with col1:
        pair = st.selectbox(
            "Crypto Pair:", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        )
    with col2:
        timeframe = st.selectbox(
            "Timeframe:", ["15m (Scalping)", "1h (Intraday)", "1d (Swing)"]
        )

    tf_value = (
        "15m" if "15m" in timeframe else ("1h" if "1h" in timeframe else "1d")
    )
    period_value = "5d" if tf_value != "1d" else "60d"

    if st.button("🤖 AI Signal & Target Fetch Karein"):
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

                    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
                    for model_name in models_to_try:
                        try:
                            res = client.models.generate_content(
                                model=model_name, contents=prompt
                            )
                            response_text = res.text
                            break
                        except Exception:
                            time.sleep(1)
                            continue

                st.markdown("---")
                st.subheader("🎯 Trade Execution Levels:")

                if response_text:
                    st.success(response_text)
                else:
                    if sma20 > sma50 and rsi < 70:
                        action = "BUY 🟢"
                        entry = f"${price:.2f}"
                        tp = f"${price * 1.015:.2f}"
                        sl = f"${price * 0.990:.2f}"
                        reason = "SMA Crossover is Bullish & RSI is safe."
                    elif rsi > 70 or sma20 < sma50:
                        action = "SELL 🔴"
                        entry = f"${price:.2f}"
                        tp = f"${price * 0.985:.2f}"
                        sl = f"${price * 1.010:.2f}"
                        reason = "RSI Overbought / Bearish SMA Crossover."
                    else:
                        action = "WAIT ⚪"
                        entry = f"${price:.2f}"
                        tp = "N/A"
                        sl = "N/A"
                        reason = "Market is consolidated."

                    st.markdown(f"### ACTION: {action}")
                    st.info(
                        f"📌 **Entry Price**: {entry}\n\n🎯 **Target Price (TP)**: {tp}\n\n🛑 **Stop-Loss (SL)**: {sl}\n\n💡 **Reason**: {reason}"
                    )
            else:
                st.error("Data fetch nahi ho pa raha hai.")

# --- TAB 2: EXTRA LARGE DAILY RRG CHART WITH CLEAR BTC ---
with tab2:
    st.subheader("📊 Ultra-HD Daily RRG Rotation Chart")
    st.write(
        "BTC ki trail line ab lambi (7-day movement) aur highlighted box me spash dikhegi."
    )

    if st.button("Generate Large HD RRG Chart"):
        with st.spinner("HD Chart plot ho raha hai..."):
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
            benchmark = "USDT-USD"

            try:
                bm_data = yf.download(
                    benchmark, period="60d", interval="1d", progress=False
                )
                if isinstance(bm_data.columns, pd.MultiIndex):
                    bm_data.columns = bm_data.columns.get_level_values(0)

                bm_close = bm_data["Close"]

                fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

                ax.axhline(100, color="gray", linestyle="--", linewidth=1.5)
                ax.axvline(100, color="gray", linestyle="--", linewidth=1.5)

                ax.text(
                    101.5,
                    101.5,
                    "LEADING 🟢\n(Strong Buy Zone)",
                    color="green",
                    fontsize=13,
                    weight="bold",
                )
                ax.text(
                    97,
                    101.5,
                    "WEAKENING 🟠\n(Take Profit Zone)",
                    color="darkorange",
                    fontsize=13,
                    weight="bold",
                )
                ax.text(
                    97,
                    98.5,
                    "LAGGING 🔴\n(Sell / Avoid Zone)",
                    color="red",
                    fontsize=13,
                    weight="bold",
                )
                ax.text(
                    101.5,
                    98.5,
                    "IMPROVING 🔵\n(Watch / Recovery Zone)",
                    color="blue",
                    fontsize=13,
                    weight="bold",
                )

                for sym in symbols:
                    s_data = yf.download(
                        sym, period="60d", interval="1d", progress=False
                    )
                    if isinstance(s_data.columns, pd.MultiIndex):
                        s_data.columns = s_data.columns.get_level_values(0)

                    s_close = s_data["Close"]

                    rs = (s_close / bm_close) * 100
                    rs_ratio = (rs / rs.rolling(14).mean()) * 100
                    rs_momentum = (rs_ratio / rs_ratio.shift(1)) * 100

                    trail_len = 7
                    x_vals = rs_ratio.iloc[-trail_len:].values
                    y_vals = rs_momentum.iloc[-trail_len:].values

                    is_btc = "BTC" in sym
                    line_color = "red" if is_btc else None
                    line_width = 3.5 if is_btc else 2.0

                    ax.plot(
                        x_vals,
                        y_vals,
                        linestyle="-",
                        alpha=0.8,
                        linewidth=line_width,
                        color=line_color,
                    )

                    ax.annotate(
                        "",
                        xy=(x_vals[-1], y_vals[-1]),
                        xytext=(x_vals[-2], y_vals[-2]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            lw=line_width + 1,
                            color="red" if is_btc else "black",
                            mutation_scale=18,
                        ),
                    )

                    box_color = "salmon" if is_btc else "yellow"
                    txt_color = "white" if is_btc else "black"

                    ax.annotate(
                        sym.replace("-USD", ""),
                        (x_vals[-1] + 0.1, y_vals[-1] + 0.1),
                        fontsize=14 if is_btc else 12,
                        weight="bold",
                        color=txt_color,
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            fc=box_color,
                            ec="black",
                            lw=1.5,
                            alpha=0.9,
                        ),
                    )

                ax.set_xlabel(
                    "JDK RS-Ratio (Daily Trend Strength)",
                    fontsize=13,
                    weight="bold",
                )
                ax.set_ylabel(
                    "JDK RS-Momentum (Daily Speed)", fontsize=13, weight="bold"
                )
                ax.set_title(
                    "Crypto Daily RRG Rotation Trajectory vs USDT",
                    fontsize=16,
                    weight="bold",
                )
                ax.grid(True, alpha=0.4, linestyle=":")

                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"RRG Chart Error: {e}")

# --- TAB 3: CHART ANALYZER ---
with tab3:
    st.subheader("📸 Screenshot Analyzer")
    uploaded_file = st.file_uploader(
        "Upload Chart Image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file and client:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

        if st.button("Analyze Image with AI"):
            with st.spinner("Analyzing chart image with AI..."):
                img_prompt = "Aap ek professional trading analyst hain. Is chart screenshot ko ache se analyze karein aur bataayein:\n1. BUY / SELL / WAIT Decision\n2. Entry Price Range\n3. Target Price (TP)\n4. Stop-Loss (SL)\n5. Key Support & Resistance Levels."

                success = False
                for m in ["gemini-2.5-flash", "gemini-2.0-flash"]:
                    try:
                        resp = client.models.generate_content(
                            model=m,
                            contents=[img, img_prompt],
                        )
                        st.success("🎯 Chart Analysis Decision:")
                        st.markdown(resp.text)
                        success = True
                        break
                    except Exception:
                        time.sleep(1)
                        continue

                if not success:
                    st.warning(
                        "⚠️ Free API Key ki Per-Minute Rate Limit Active Hai. Kripya 30-40 second ka pause lein aur dubara button press karein."
                    )
