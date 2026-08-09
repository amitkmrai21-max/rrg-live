import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import time
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="AI Trading Assistant", layout="centered")

# Streamlit Secrets se API key automatic padhega
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    client = None
    st.error("API Key secrets me set nahi hai!")

st.title("AI Trading Assistant")

tab1, tab2 = st.tabs(["📸 Chart Analyzer", "📡 Live Signals"])

# ==================== TAB 1: Screenshot Analyzer (existing) ====================
with tab1:
    st.write("Upload chart screenshot for Buy/Sell signals")
    uploaded_file = st.file_uploader("Upload Chart Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        max_width = 1024
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)))
        st.image(image, caption='Uploaded Chart', use_container_width=True)

        if st.button('Analyze Chart'):
            prompt = """
            You are an expert technical analyst. Analyze this chart image:
            1. Current Trend & Patterns identified
            2. Signal: BUY / SELL / WAIT
            3. Entry Price, Target Price, and Stop Loss
            4. Risk to Reward Ratio
            Keep it clear and precise.
            """
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with st.spinner(f"Analyzing... (attempt {attempt + 1}/{max_retries})"):
                        response = client.models.generate_content(
                            model='gemini-3.5-flash',
                            contents=[prompt, image],
                            config=types.GenerateContentConfig(
                                http_options=types.HttpOptions(timeout=60000)
                            ),
                        )
                    st.subheader("Analysis & Signal:")
                    st.write(response.text)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Server busy hai, retry ho raha hai... ({attempt + 1}/{max_retries})")
                        time.sleep(3)
                    else:
                        st.error("Gemini server abhi available nahi hai. Thodi der baad try karo.")
                        st.exception(e)

# ==================== TAB 2: Live Signals (new) ====================

def fetch_klines(symbol, interval, limit=100):
    """Binance public API — no key chahiye, free candlestick data."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    candles = []
    for k in raw:
        candles.append({
            "time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
        })
    return candles


def compute_signal(candles):
    closes = [c["close"] for c in candles]
    n = len(closes)
    if n < 51:
        return None

    def sma(period, end_idx):
        return sum(closes[end_idx - period + 1:end_idx + 1]) / period

    sma20_now, sma50_now = sma(20, n - 1), sma(50, n - 1)
    sma20_prev, sma50_prev = sma(20, n - 2), sma(50, n - 2)

    gains = losses = 0
    for i in range(n - 14, n):
        diff = closes[i] - closes[i - 1]
        gains += max(diff, 0)
        losses += max(-diff, 0)
    avg_gain, avg_loss = gains / 14, losses / 14
    rsi = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    ranges = [c["high"] - c["low"] for c in candles[-14:]]
    avg_range = sum(ranges) / len(ranges)
    price = closes[-1]

    crossed_up = sma20_prev <= sma50_prev and sma20_now > sma50_now
    crossed_down = sma20_prev >= sma50_prev and sma20_now < sma50_now

    if crossed_up and rsi < 70:
        signal, reason = "BUY", "SMA20 abhi SMA50 ke upar cross hua — fresh bullish momentum"
    elif crossed_down and rsi > 30:
        signal, reason = "SELL", "SMA20 abhi SMA50 ke neeche cross hua — fresh bearish momentum"
    elif sma20_now > sma50_now and rsi < 70:
        signal, reason = "BUY", "Uptrend chal raha hai (SMA20 > SMA50), RSI overbought nahi"
    elif sma20_now < sma50_now and rsi > 30:
        signal, reason = "SELL", "Downtrend chal raha hai (SMA20 < SMA50), RSI oversold nahi"
    else:
        signal, reason = "WAIT", "Trend clear nahi ya RSI extreme zone me hai"

    if signal == "BUY":
        target, stop = price + avg_range * 3, price - avg_range * 1.5
    elif signal == "SELL":
        target, stop = price - avg_range * 3, price + avg_range * 1.5
    else:
        target = stop = None

    return {
        "price": price, "signal": signal, "reason": reason, "rsi": rsi,
        "target": target, "stop": stop, "time": candles[-1]["time"],
    }


def render_card(symbol, interval_label, result):
    if result is None:
        st.warning(f"{symbol} ({interval_label}): abhi data kaafi nahi hai")
        return
    icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡"}[result["signal"]]
    with st.container(border=True):
        st.markdown(f"**{icon} {symbol} · {interval_label} · {result['signal']}**")
        st.caption(result["reason"])
        c1, c2 = st.columns(2)
        c1.metric("Price", f"${result['price']:,.2f}")
        c2.metric("RSI", f"{result['rsi']:.0f}")
        if result["target"]:
            c3, c4 = st.columns(2)
            c3.metric("Target", f"${result['target']:,.2f}")
            c4.metric("Stop Loss", f"${result['stop']:,.2f}")
        st.caption(f"Updated: {result['time'].strftime('%H:%M UTC')}")


with tab2:
    st.write("BTC/USDT aur ETH/USDT ke live signals — real price data pe based (SMA crossover + RSI)")
    refresh_seconds = st.selectbox(
        "Refresh interval", [900, 1800, 3600],
        format_func=lambda s: f"Har {s // 60} minute", index=0
    )

    @st.fragment(run_every=refresh_seconds)
    def live_signals():
        st.caption(f"Last checked: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            cols = st.columns(2)
            for col, interval, label in zip(cols, ["15m", "1h"], ["15 min", "1 hour"]):
                with col:
                    try:
                        candles = fetch_klines(symbol, interval, limit=100)
                        result = compute_signal(candles)
                        render_card(symbol.replace("USDT", "/USDT"), label, result)
                    except Exception as e:
                        st.error(f"{symbol} {label}: fetch error")

    live_signals()

    st.caption("⚠️ Ye rule-based technical signals hain, financial advice nahi. Apna risk khud manage karo.")
