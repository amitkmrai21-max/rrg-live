import io
import time
import google.generativeai as genai_old
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
import yfinance as yf
from datetime import datetime

# Streamlit Page Setup
st.set_page_config(
    page_title="Smart Trade AI",
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

# API Setup with google-generativeai (Legacy SDK for flawless Vision)
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    genai_old.configure(api_key=api_key)
else:
    st.error("⚠️ GEMINI_API_KEY Streamlit Secrets me set nahi hai!")

st.title("🚀 Smart Trade AI: Assistant & Paper Trading")

# Initialize Session State for Paper Trading
if 'balance' not in st.session_state:
    st.session_state.balance = 10000.0  # Initial Virtual Balance
if 'positions' not in st.session_state:
    st.session_state.positions = []     # Open Trades List

# Defined Crypto Symbols for the App
crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


# Helper Function: Robust Fetch Real-Time Data (Handle Rate Limits)
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

                # Technical Indicators
                df["SMA20"] = df["Close"].rolling(20).mean()
                df["SMA50"] = df["Close"].rolling(50).mean()

                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                return df
        except Exception as e:
            if "RateLimitError" in str(e) or 429 in str(e):
                time.sleep(2) # Wait if rate limited
                continue
            return None
    return None

def get_current_price(symbol, retries=2):
    ticker_sym = symbol.replace("USDT", "-USD")
    for attempt in range(retries + 1):
        try:
            ticker = yf.Ticker(ticker_sym)
            todays_data = ticker.history(period='1d')
            if not todays_data.empty:
                return todays_data['Close'].iloc[-1]
        except Exception as e:
             if "RateLimitError" in str(e) or 429 in str(e):
                time.sleep(2) # Wait if rate limited
                continue
             return None
    return None

# Defining Tabs
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
        pair = st.selectbox(
            "Crypto Pair:", crypto_symbols, key="signal_pair"
        )
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
                if api_key:
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
                    try:
                        model = genai_old.GenerativeModel("gemini-1.5-flash")
                        res = model.generate_content(prompt)
                        response_text = res.text
                    except Exception:
                        response_text = None

                st.markdown("---")
                st.subheader("🎯 Trade Execution Levels:")

                if response_text:
                    st.success(response_text)
                else:
                    st.info("🤖 AI generated signals currently unavailable. Check technical indicators above.")
            else:
                st.error("⚠️ Yahoo Finance Rate Limit active hai. Kripya 5-10 second baad dobara koshish karein.")

# --- TAB 2:💼 LIVE PAPER TRADING (Virtual Trading Simulator) ---
with tab2:
    st.subheader("💼 Live Paper Trading Simulator")
    st.write("Real-time market prices par dummy money ke saath trading practice karein.")

    # Section 1: Portfolio Summary
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Virtual Balance", f"${st.session_state.balance:,.2f}")
    
    # Calculate Total P&L live
    current_pnl = 0.0
    for pos in st.session_state.positions:
        current_price = get_current_price(pos['symbol'])
        if current_price is not None:
            if pos['type'] == 'BUY':
                pnl = (current_price - pos['entry_price']) * pos['amount']
            else: # SELL
                pnl = (pos['entry_price'] - current_price) * pos['amount']
            current_pnl += pnl
            pos['live_pnl'] = pnl # Update live P&L in the list for display

    with c2:
        pnl_color = "green" if current_pnl >= 0 else "red"
        st.markdown(f"#### Live Portfolio P&L: <span style='color:{pnl_color};'>${current_pnl:,.2f}</span>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Section 2: Place New Trade
    st.write("#### ➕ Place New Trade")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        trade_pair = st.selectbox("Select Crypto:", crypto_symbols, key="trade_pair")
    with col2:
        trade_type = st.selectbox("Type:", ["BUY", "SELL"], key="trade_type")
    
    # Section to Fetch live price (Handling rate limit error message here)
    try:
        live_entry_price = get_current_price(trade_pair)
        with col3:
            if live_entry_price:
                st.write(f"**Current Price:**\n ${live_entry_price:,.2f}")
            else:
                st.write("**Current Price:**\n N/A (Rate Limit)")
    except Exception:
        live_entry_price = None
        with col3:
            st.write("**Current Price:**\n API Waiting...")
            
    with col4:
        amount_usd = st.number_input("Amount (USD):", min_value=10.0, step=10.0, value=100.0)
    
    if st.button(f"Place {trade_type} Order", key="place_order_btn"):
        if not live_entry_price:
            st.warning("⚠️ Live Price available nahi hai. Yahoo Finance API limited hai, kripya refresh karein.")
        else:
            # Check balance
            if amount_usd > st.session_state.balance:
                st.error("Insufficient virtual balance!")
            else:
                st.session_state.balance -= amount_usd
                
                new_position = {
                    'id': len(st.session_state.positions) + 1,
                    'symbol': trade_pair,
                    'type': trade_type,
                    'amount_usd': amount_usd,
                    'amount': amount_usd / live_entry_price, # crypto amount
                    'entry_price': live_entry_price,
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'live_pnl': 0.0
                }
                st.session_state.positions.append(new_position)
                st.success(f"{trade_type} order for ${amount_usd} placed successfully. Position open.")
                time.sleep(1)
                st.rerun()

    st.markdown("---")

    # Section 3: Open Positions (Live Tracking)
    st.write("#### 📂 Open Positions (Real-time tracking)")
    if not st.session_state.positions:
        st.info("No open positions. Use the section above to place a trade.")
    else:
        # Create a display DataFrame
        display_positions = []
        for pos in st.session_state.positions:
            display_positions.append({
                'ID': pos['id'],
                'Time': pos['time'],
                'Pair': pos['symbol'],
                'Type': pos['type'],
                'Amt (USD)': f"${pos['amount_usd']:,.2f}",
                'Entry Price': f"${pos['entry_price']:,.2f}",
                'Live P&L': f"${pos['live_pnl']:,.2f}"
            })
        
        df_display = pd.DataFrame(display_positions)
        st.dataframe(df_display, use_container_width=True)

        # Close Position section
        st.write("#### ❌ Close Position")
        col_close1, col_close2 = st.columns([3, 1])
        with col_close1:
            pos_to_close_id = st.selectbox("Select Position ID to close:", df_display['ID'].tolist())
        
        if st.button("Close Selected Position", key="close_pos_btn"):
            # Find the position
            position_found = False
            for i, pos in enumerate(st.session_state.positions):
                if pos['id'] == pos_to_close_id:
                    position_found = True
                    # Fetch latest price to settle
                    try:
                        close_price = get_current_price(pos['symbol'])
                        if close_price:
                            # Calculate final P&L
                            if pos['type'] == 'BUY':
                                final_pnl = (close_price - pos['entry_price']) * pos['amount']
                            else: # SELL
                                final_pnl = (pos['entry_price'] - close_price) * pos['amount']
                            
                            # Add back principle + P&L to balance
                            st.session_state.balance += (pos['amount_usd'] + final_pnl)
                            
                            # Remove from open positions
                            st.session_state.positions.pop(i)
                            st.success(f"Position ID {pos_to_close_id} closed successfully.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("⚠️ Settle price nahi mil pa raha. Yahoo Finance key limited hai, 5-10 sec baad refresh karein.")
                    except Exception:
                        st.error("⚠️ Error closing position. Rate limit hit.")
                    break
            if not position_found:
                 st.error("Position ID not found.")
        
        if st.button("Refresh Live Prices & Portfolio", key="refresh_pnl_btn"):
            with st.spinner("Refreshing Prices..."):
                time.sleep(1)
                st.rerun()

# --- TAB 3: EXTRA LARGE DAILY RRG CHART WITH CLEAR BTC ---
with tab3:
    st.subheader("📊 Ultra-HD Daily RRG Rotation Chart")
    st.write(
        "BTC ki trail line ab lambi (7-day movement) aur highlighted box me spash dikhegi."
    )

    if st.button("Generate Large HD RRG Chart", key="gen_rrg_btn"):
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

# --- TAB 4: CHART ANALYZER ---
with tab4:
    st.subheader("📸 Screenshot Analyzer")
    uploaded_file = st.file_uploader(
        "Upload Chart Image", type=["jpg", "jpeg", "png"], key="chart_uploader"
    )
    if uploaded_file and api_key:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

        if st.button("Analyze Image with AI", key="analyze_image_btn"):
            with st.spinner("Analyzing chart image with Gemini Vision..."):
                try:
                    img_prompt = "Aap ek professional trading analyst hain. Is chart screenshot ko ache se analyze karein aur bataayein:\n1. BUY / SELL / WAIT Decision\n2. Entry Price Range\n3. Target Price (TP)\n4. Stop-Loss (SL)\n5. Key Support & Resistance Levels."

                    model_v = genai_old.GenerativeModel("gemini-1.5-flash")
                    response = model_v.generate_content([img_prompt, img])

                    st.success("🎯 Chart Analysis Decision:")
                    st.markdown(response.text)

                except Exception as err:
                    st.error(f"Analysis Error: {err}. Free API key often hits quota limits; try again after 1 minute.")
