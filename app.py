import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from google import genai

# Streamlit Page Setup
st.set_page_config(page_title="AI Trading Assistant", layout="wide")

# Gemini API Initialization
client = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("⚠️ GEMINI_API_KEY Streamlit Secrets me set nahi hai!")

st.title("🚀 AI Trading Assistant & RRG Rotation")

tab1, tab2, tab3 = st.tabs(["🤖 Real-Time AI Signals", "📊 Daily RRG Chart (HD)", "📸 Chart Analyzer"])

# Helper Function: Fetch Real-Time Data via yfinance
def get_crypto_data(symbol, period="30d", interval="1d"):
    ticker = symbol.replace("USDT", "-USD")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Technical Indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    except Exception:
        return None

# --- TAB 1: REAL-TIME AI SIGNALS ---
with tab1:
    st.subheader("💡 AI Live Buy / Sell / Target Signals")
    
    col1, col2 = st.columns(2)
    with col1:
        pair = st.selectbox("Crypto Pair:", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    with col2:
        timeframe = st.selectbox("Timeframe:", ["15m (Scalping)", "1h (Intraday)", "1d (Swing)"])
    
    tf_value = "15m" if "15m" in timeframe else ("1h" if "1h" in timeframe else "1d")
    period_value = "5d" if tf_value != "1d" else "60d"
    
    if st.button("🤖 AI Signal & Target Fetch Karein"):
        with st.spinner("Market Data & AI Analysis chal raha hai..."):
            df = get_crypto_data(pair, period=period_value, interval=tf_value)
            if df is not None and not df.empty:
                price = float(df['Close'].iloc[-1])
                rsi = float(df['RSI'].iloc[-1])
                sma20 = float(df['SMA20'].iloc[-1]) if 'SMA20' in df else price
                sma50 = float(df['SMA50'].iloc[-1]) if 'SMA50' in df else price
                
                st.write("### Live Indicators:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${price:.2f}")
                c2.metric("RSI (14)", f"{rsi:.1f}")
                c3.metric("Trend", "Bullish 🟢" if sma20 > sma50 else "Bearish 🔴")
                
                if client:
                    try:
                        prompt = f"""
                        Aap ek expert crypto analyst hain. Niche diye gaye real-time data par analysis karein:
                        - Pair: {pair}
                        - Timeframe: {tf_value}
                        - Current Price: ${price:.2f}
                        - RSI: {rsi:.1f}
                        - SMA20: ${sma20:.2f} | SMA50: ${sma50:.2f}
                        
                        Kripya neeche diye gaye FORMAT me hi response dein:
                        
                        ### 🔴/🟢 ACTION: [BUY / SELL / WAIT]
                        - **Entry Price**: [Specific Price Range]
                        - **Target Price (TP)**: [Target Value]
                        - **Stop-Loss (SL)**: [Safety Level]
                        - **Reasoning**: [1-2 lines clear logic]
                        """
                        # Corrected Model ID for New Google GenAI Library
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt
                        )
                        st.markdown("---")
                        st.subheader("🎯 Gemini AI Signal Output:")
                        st.success(response.text)
                    except Exception as e:
                        st.error(f"AI Signal Error: {e}")
                else:
                    st.error("API Key missing hai!")
            else:
                st.error("Data fetch nahi ho pa raha hai.")

# --- TAB 2: LARGE & CLEAR DAILY RRG CHART ---
with tab2:
    st.subheader("📊 High-Resolution Daily Relative Rotation Graph")
    st.write("Bada view aur spash teer (arrows) directional rotation samajhne ke liye.")
    
    if st.button("Generate HD RRG Chart"):
        with st.spinner("HD Daily RRG Chart generate ho raha hai..."):
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
            benchmark = "USDT-USD"
            
            try:
                bm_data = yf.download(benchmark, period="60d", interval="1d", progress=False)
                if isinstance(bm_data.columns, pd.MultiIndex):
                    bm_data.columns = bm_data.columns.get_level_values(0)
                
                bm_close = bm_data['Close']
                
                # High Resolution & Larger Canvas Size (12x9)
                fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
                
                # Quadrants
                ax.axhline(100, color='gray', linestyle='--', linewidth=1.5)
                ax.axvline(100, color='gray', linestyle='--', linewidth=1.5)
                
                ax.text(102, 102, "LEADING 🟢\n(Strong Buy / Upar Uth Raha Hai)", color='green', fontsize=12, weight='bold')
                ax.text(96.5, 102, "WEAKENING 🟠\n(Take Profit / Weak Ho Raha)", color='darkorange', fontsize=12, weight='bold')
                ax.text(96.5, 98, "LAGGING 🔴\n(Sell / Gir Raha Hai)", color='red', fontsize=12, weight='bold')
                ax.text(102, 98, "IMPROVING 🔵\n(Watch / Recovery Zone)", color='blue', fontsize=12, weight='bold')
                
                for sym in symbols:
                    s_data = yf.download(sym, period="60d", interval="1d", progress=False)
                    if isinstance(s_data.columns, pd.MultiIndex):
                        s_data.columns = s_data.columns.get_level_values(0)
                        
                    s_close = s_data['Close']
                    
                    rs = (s_close / bm_close) * 100
                    rs_ratio = (rs / rs.rolling(14).mean()) * 100
                    rs_momentum = (rs_ratio / rs_ratio.shift(1)) * 100
                    
                    x_vals = rs_ratio.iloc[-4:].values
                    y_vals = rs_momentum.iloc[-4:].values
                    
                    # Thicker Trail Plot
                    ax.plot(x_vals, y_vals, linestyle='-', alpha=0.6, linewidth=2.5)
                    
                    # Bada Arrow Head
                    ax.annotate(
                        "", 
                        xy=(x_vals[-1], y_vals[-1]), 
                        xytext=(x_vals[-2], y_vals[-2]),
                        arrowprops=dict(arrowstyle="-|>", lw=3, color='black', mutation_scale=15)
                    )
                    
                    # Clear Bold Symbol Label
                    ax.annotate(
                        sym.replace("-USD", ""), 
                        (x_vals[-1] + 0.15, y_vals[-1] + 0.15), 
                        fontsize=13, 
                        weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.3)
                    )
                    
                ax.set_xlabel("JDK RS-Ratio (Daily Trend Strength)", fontsize=12, weight='bold')
                ax.set_ylabel("JDK RS-Momentum (Daily Speed)", fontsize=12, weight='bold')
                ax.set_title("Crypto Daily RRG Rotation Trajectory vs USDT", fontsize=15, weight='bold')
                ax.grid(True, alpha=0.4, linestyle=':')
                
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"RRG Chart Error: {e}")

# --- TAB 3: CHART ANALYZER ---
with tab3:
    st.subheader("📸 Screenshot Analyzer")
    uploaded_file = st.file_uploader("Upload Chart Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file and client:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
        if st.button("Analyze Image with AI"):
            with st.spinner("Analyzing image..."):
                try:
                    resp = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[img, "Provide Buy/Sell decision, Entry, Target and Stoploss for this chart."]
                    )
                    st.write(resp.text)
                except Exception as e:
                    st.error(f"Image Analysis Error: {e}")
