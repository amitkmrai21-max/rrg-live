from datetime import datetime, timezone
import time
from google import genai
from google.genai import types
import pandas as pd
from PIL import Image
import streamlit as st
import yfinance as yf

# Page Config
st.set_page_config(
    page_title="AI Trading Assistant", layout="centered"
)

# API Key handling
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    client = None
    st.error("API Key secrets me set nahi hai!")

st.title("AI Trading Assistant")

tab1, tab2 = st.tabs(["📸 Chart Analyzer", "📡 Live Signals"])

# --- TAB 1: CHART ANALYZER ---
with tab1:
    st.subheader("Upload chart screenshot for Buy/Sell signals")
    uploaded_file = st.file_uploader(
        "Upload Chart Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chart", use_container_width=True)

        if st.button("Analyze Chart with AI"):
            if client is None:
                st.error("API Key missing! Pehle secrets me API key dalein.")
            else:
                with st.spinner("Chart analyze ho raha hai..."):
                    try:
                        prompt = "Analyze this trading chart. Give Buy/Sell signal, Key Support/Resistance, and Risk Level in clear, concise bullet points."
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=[image, prompt]
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")

# --- TAB 2: LIVE SIGNALS ---
with tab2:
    st.write(
        "BTC/USDT aur ETH/USDT ke live signals — real price data pe based (SMA crossover + RSI)"
    )

    refresh_option = st.selectbox(
        "Refresh interval", ["Har 15 minute", "Har 1 hour"]
    )

    # Fetch Data Function using yfinance (Bypasses Binance Block)
    def fetch_data(symbol, interval_str):
        ticker = symbol.replace("USDT", "-USD")
        interval = "15m" if "15" in interval_str else "1h"
        try:
            df = yf.download(
                ticker, period="5d", interval=interval, progress=False
            )
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Signal Logic (SMA + RSI)
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()

            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            return df
        except:
            return None

    # Render Signals
    pairs = [
        ("BTCUSDT", "15 min"),
        ("BTCUSDT", "1 hour"),
        ("ETHUSDT", "15 min"),
        ("ETHUSDT", "1 hour"),
    ]

    for symbol, tf in pairs:
        df = fetch_data(symbol, tf)
        if df is not None and not df.empty:
            close_p = float(df["Close"].iloc[-1])
            rsi_val = float(df["RSI"].iloc[-1])
            sma20 = float(df["SMA20"].iloc[-1])
            sma50 = float(df["SMA50"].iloc[-1])

            if sma20 > sma50 and rsi_val < 70:
                sig = "BUY 🟢"
            elif sma20 < sma50 and rsi_val > 30:
                sig = "SELL 🔴"
            else:
                sig = "NEUTRAL ⚪"

            st.success(
                f"**{symbol} {tf}:** Signal: **{sig}** | Price: **${close_p:.2f}** | RSI: **{rsi_val:.1f}**"
            )
        else:
            st.error(f"**{symbol} {tf}:** fetch error")

    st.warning(
        "⚠️ Ye rule-based technical signals hain, financial advice nahi. Apna risk khud manage karo."
    )
