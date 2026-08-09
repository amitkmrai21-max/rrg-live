  from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from google import genai

# Page Config
st.set_page_config(page_title="AI Trading Assistant", layout="wide")

# Gemini API Client Setup
client = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error(
        "⚠️ API Key secrets me missing hai! (GEMINI_API_KEY set karein)."
    )

st.title("🚀 AI Real-Time Trading Assistant & RRG")

tab1, tab2, tab3 = st.tabs(
    ["🤖 Real-Time AI Signals", "📊 RRG Rotation Chart", "📸 Chart Analyzer"]
)


# --- DATA FETCH FUNCTION ---
def get_crypto_data(symbol, period="5d", interval="15m"):
    ticker = symbol.replace("USDT", "-USD")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
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
    st.subheader("💡 AI-Powered Live Buy/Sell Decisions")
    pair = st.selectbox(
        "Crypto Pair Select Karein:",
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
    )
    timeframe = st.selectbox(
        "Timeframe Select Karein:", ["15m (Scalping)", "1h (Intraday)"]
    )

    tf_value = "15m" if "15m" in timeframe else "1h"

    if st.button("🤖 AI Signal Decision Fetch Karein"):
        with st.spinner("Real-time market data & AI Analysis chal raha hai..."):
            df = get_crypto_data(pair, interval=tf_value)
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                rsi = float(df["RSI"].iloc[-1])
                sma20 = float(df["SMA20"].iloc[-1])
                sma50 = float(df["SMA50"].iloc[-1])

                # Data summary for AI
                market_summary = f"""
                Crypto Pair: {pair}
                Timeframe: {tf_value}
                Current Price: ${price:.2f}
                RSI (14): {rsi:.1f}
                SMA 20: ${sma20:.2f}
                SMA 50: ${sma50:.2f}
                Trend: {'Bullish' if sma20 > sma50 else 'Bearish'}
                """

                st.write("### Market Data Metrics:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Live Price", f"${price:.2f}")
                c2.metric("RSI (14)", f"{rsi:.1f}")
                c3.metric(
                    "Trend", "Bullish 🟢" if sma20 > sma50 else "Bearish 🔴"
                )

                # AI Decision Generation
                if client:
                    try:
                        prompt = f"""
                        Aap ek professional Crypto Trader hain. Niche diye gaye live market data ke basis par trading decision dein:
                        {market_summary}
                        
                        Muje spash (clear) format me batayein:
                        1. **DECISION**: [BUY / SELL / WAIT]
                        2. **ENTRY PRICE**: [Ideal Range]
                        3. **TARGET (TP)**: [Target Price]
                        4. **STOP-LOSS (SL)**: [Safety Level]
                        5. **REASON**: [2 short lines me justification]
                        """
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=prompt
                        )
                        st.markdown("---")
                        st.subheader("🎯 Gemini AI Trading Advice:")
                        st.success(response.text)
                    except Exception as e:
                        st.error(f"AI Signal Error: {e}")
                else:
                    st.error("Gemini Client initialize nahi hua.")
            else:
                st.error("Market data fetch nahi ho paya.")

# --- TAB 2: REAL-TIME RRG CHART ---
with tab2:
    st.subheader("🌐 Relative Rotation Graph (RRG) - Market Momentum")
    st.info(
        "RRG quadrant chart se pata chalta hai kaunsa coin Leading (Strong) hai aur kaunsa Lagging (Weak) benchmark (BTC) ke mukable."
    )

    if st.button("Generate Live RRG Chart"):
        with st.spinner("Coins rotation calculate ho raha hai..."):
            symbols = ["ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
            benchmark = "BTC-USD"

            try:
                # Fetch Benchmark Data
                bm_data = yf.download(
                    benchmark, period="1mo", interval="1d", progress=False
                )
                if isinstance(bm_data.columns, pd.MultiIndex):
                    bm_data.columns = bm_data.columns.get_level_values(0)

                bm_close = bm_data["Close"]

                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot Quadrants
                ax.axhline(100, color="gray", linestyle="--")
                ax.axvline(100, color="gray", linestyle="--")

                # Quadrant Labels
                ax.text(
                    102,
                    102,
                    "LEADING (Strong Buy)",
                    color="green",
                    fontsize=10,
                    weight="bold",
                )
                ax.text(
                    96,
                    102,
                    "WEAKENING (Take Profit)",
                    color="orange",
                    fontsize=10,
                    weight="bold",
                )
                ax.text(
                    96,
                    98,
                    "LAGGING (Avoid/Sell)",
                    color="red",
                    fontsize=10,
                    weight="bold",
                )
                ax.text(
                    102,
                    98,
                    "IMPROVING (Watch/Buy)",
                    color="blue",
                    fontsize=10,
                    weight="bold",
                )

                for sym in symbols:
                    s_data = yf.download(
                        sym, period="1mo", interval="1d", progress=False
                    )
                    if isinstance(s_data.columns, pd.MultiIndex):
                        s_data.columns = s_data.columns.get_level_values(0)

                    s_close = s_data["Close"]

                    # Relative Strength Ratio & Momentum Calculation
                    rs = (s_close / bm_close) * 100
                    rs_ratio = (rs / rs.rolling(14).mean()) * 100
                    rs_momentum = (
                        rs_ratio / rs_ratio.shift(1)
                    ) * 100  # Momentum

                    x = float(rs_ratio.iloc[-1])
                    y = float(rs_momentum.iloc[-1])

                    ax.scatter(x, y, s=120)
                    ax.annotate(
                        sym.replace("-USD", ""),
                        (x + 0.2, y + 0.2),
                        fontsize=11,
                        weight="bold",
                    )

                ax.set_xlabel("JDK RS-Ratio (Trend Strength)")
                ax.set_ylabel("JDK RS-Momentum (Speed)")
                ax.set_title("Crypto RRG Rotation vs BTC Benchmark")
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
            except Exception as e:
                st.error(f"RRG Chart calculation error: {e}")

# --- TAB 3: CHART ANALYZER ---
with tab3:
    st.subheader("📸 Screenshot Analyzer")
    uploaded_file = st.file_uploader(
        "Upload Chart Image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file and client:
        from PIL import Image

        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        img,
                        "Provide Buy/Sell signal, Target and Stoploss for this chart.",
                    ],
                )
                st.write(resp.text)
