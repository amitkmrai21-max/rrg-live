"""
Relative Rotation Graph (RRG) - Live Web App
NSE, BSE, Commodity aur Crypto - char jagah kaam karega
Free hosting: Streamlit Community Cloud
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="RRG - Relative Rotation Graph", layout="wide")

st.title("📊 Relative Rotation Graph (RRG)")
st.caption("Data source: Yahoo Finance (free, thoda delay ho sakta hai)")

# ---------------- SIDEBAR CONFIG ----------------

st.sidebar.header("Settings")

MARKET = st.sidebar.selectbox("Market chuno", ["NSE", "BSE", "COMMODITY", "CRYPTO"])

if MARKET == "NSE":
    BENCHMARK = "^NSEI"
    DEFAULT_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                        "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
                        "LT.NS", "KOTAKBANK.NS"]

elif MARKET == "BSE":
    BENCHMARK = "^BSESN"
    DEFAULT_SYMBOLS = ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO",
                        "ICICIBANK.BO"]

elif MARKET == "COMMODITY":
    BENCHMARK = "GC=F"
    DEFAULT_SYMBOLS = ["SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "ZC=F", "KC=F"]

elif MARKET == "CRYPTO":
    BENCHMARK = "BTC-USD"
    DEFAULT_SYMBOLS = ["ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
                        "ADA-USD", "DOGE-USD"]

symbols_text = st.sidebar.text_area(
    "Symbols (comma se alag karo)",
    value=", ".join(DEFAULT_SYMBOLS),
    height=100
)
SYMBOLS = [s.strip() for s in symbols_text.split(",") if s.strip()]

PERIOD = st.sidebar.selectbox("Period", ["1y", "2y", "3y"], index=1)
INTERVAL = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
RS_WINDOW = st.sidebar.slider("Smoothing window", 5, 20, 10)
TAIL_LENGTH = st.sidebar.slider("Tail length", 3, 15, 6)

run_button = st.sidebar.button("🔄 Refresh / Run", type="primary")

# ---------------- FUNCTIONS ----------------


@st.cache_data(ttl=3600)  # 1 hour cache - rate limit se bachne ke liye
def get_data(symbols, benchmark, period, interval):
    import time
    tickers = symbols + [benchmark]
    last_error = None
    for attempt in range(3):
        try:
            data = yf.download(tickers, period=period, interval=interval,
                                progress=False, threads=False)["Close"]
            data = data.dropna(how="all")
            if not data.empty:
                return data
        except Exception as e:
            last_error = e
        time.sleep(5)  # thoda ruk kar retry karo
    if last_error:
        raise last_error
    return pd.DataFrame()


def compute_rrg(data, symbols, benchmark, window):
    rs_ratio_dict = {}
    rs_mom_dict = {}
    bench = data[benchmark]

    for sym in symbols:
        if sym not in data.columns:
            continue
        rs = (data[sym] / bench) * 100
        rs_mean = rs.rolling(window).mean()
        rs_std = rs.rolling(window).std()
        rs_ratio = 100 + ((rs - rs_mean) / rs_std)

        rs_ratio_smooth = rs_ratio.rolling(3).mean()
        rs_mom = 100 + ((rs_ratio_smooth - rs_ratio_smooth.shift(window))
                         / rs_ratio_smooth.shift(window) * 100)

        rs_ratio_dict[sym] = rs_ratio_smooth
        rs_mom_dict[sym] = rs_mom

    rs_ratio_df = pd.DataFrame(rs_ratio_dict).dropna(how="all")
    rs_mom_df = pd.DataFrame(rs_mom_dict).dropna(how="all")
    return rs_ratio_df, rs_mom_df


def plot_rrg(rs_ratio_df, rs_mom_df, tail=6):
    common_idx = rs_ratio_df.index.intersection(rs_mom_df.index)
    rs_ratio_df = rs_ratio_df.loc[common_idx]
    rs_mom_df = rs_mom_df.loc[common_idx]

    fig = go.Figure()

    fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115,
                  fillcolor="lightgreen", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=85, y0=100, x1=100, y1=115,
                  fillcolor="lightblue", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=85, y0=85, x1=100, y1=100,
                  fillcolor="lightcoral", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=100, y0=85, x1=115, y1=100,
                  fillcolor="khaki", opacity=0.25, line_width=0)

    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.add_vline(x=100, line_dash="dot", line_color="gray")

    colors = ["red", "blue", "green", "orange", "purple",
              "brown", "magenta", "cyan", "black", "teal"]

    for i, sym in enumerate(rs_ratio_df.columns):
        x = rs_ratio_df[sym].dropna().tail(tail)
        y = rs_mom_df[sym].dropna().tail(tail)
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]

        if len(x) == 0:
            continue

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=5),
            name=sym, showlegend=False
        ))

        label = sym.replace(".NS", "").replace(".BO", "").replace("-USD", "").replace("=F", "")
        fig.add_trace(go.Scatter(
            x=[x.iloc[-1]], y=[y.iloc[-1]],
            mode="markers+text",
            marker=dict(size=12, color=color),
            text=[label],
            textposition="top center",
            name=sym
        ))

    fig.update_layout(
        xaxis_title="JdK RS-Ratio",
        yaxis_title="JdK RS-Momentum",
        height=750,
        plot_bgcolor="white"
    )

    fig.add_annotation(x=112, y=112, text="LEADING", showarrow=False,
                        font=dict(color="green", size=14))
    fig.add_annotation(x=88, y=112, text="IMPROVING", showarrow=False,
                        font=dict(color="blue", size=14))
    fig.add_annotation(x=88, y=88, text="LAGGING", showarrow=False,
                        font=dict(color="red", size=14))
    fig.add_annotation(x=112, y=88, text="WEAKENING", showarrow=False,
                        font=dict(color="orange", size=14))

    return fig


# ---------------- MAIN ----------------

with st.spinner(f"{MARKET} data fetch ho raha hai..."):
    try:
        data = get_data(SYMBOLS, BENCHMARK, PERIOD, INTERVAL)
        rs_ratio_df, rs_mom_df = compute_rrg(data, SYMBOLS, BENCHMARK, RS_WINDOW)

        if rs_ratio_df.empty or rs_mom_df.empty:
            st.warning("Data kaafi nahi hai. Period badha do (2y ya 3y try karo).")
        else:
            fig = plot_rrg(rs_ratio_df, rs_mom_df, tail=TAIL_LENGTH)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Last updated: {data.index[-1].strftime('%d-%b-%Y')} | Benchmark: {BENCHMARK}")
    except Exception as e:
        st.error(f"Error aaya: {e}")
