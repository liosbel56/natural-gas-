import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# Kalshi Signal Bot — versión corregida
# - No usa streamlit_autorefresh (evita el error del componente)
# - Actualiza con st.fragment(run_every=...)
# - Señales visibles: BUY UP / BUY DOWN / WAIT
# - BTC/USD: velas reales de Coinbase Exchange
# - EUR/USD: Yahoo Finance (puede tener retraso; no es TradingView)
# - Incluye ciclo visual de 15 minutos y fondo morado
# ============================================================

st.set_page_config(
    page_title="Kalshi Signal Bot",
    page_icon="📈",
    layout="wide",
)

REFRESH_SECONDS = 10
CANDLE_SECONDS = 60
CANDLE_LIMIT = 200
MIN_CANDLES = 55

COINBASE_API = "https://api.exchange.coinbase.com"
YAHOO_API = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"
KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"

ASSETS = {
    "BTC/USD": {"source": "Coinbase Exchange", "symbol": "BTC-USD"},
    "EUR/USD": {"source": "Yahoo Finance", "symbol": "EURUSD=X"},
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; MarketSignalBot/1.0)",
    "Accept": "application/json",
})

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #10051f 0%, #1d0b36 45%, #2b1050 100%);
        color: #f8f5ff;
    }
    [data-testid="stHeader"] { background: rgba(16, 5, 31, 0.85); }
    [data-testid="stSidebar"] { background: #17082a; }
    h1, h2, h3, p, label, .stMarkdown { color: #f8f5ff; }
    .signal-card {
        padding: 18px 20px; border-radius: 16px; margin: 8px 0 16px 0;
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.15);
    }
    .signal-up { color: #5df59b !important; font-size: 2rem; font-weight: 800; }
    .signal-down { color: #ff7777 !important; font-size: 2rem; font-weight: 800; }
    .signal-wait { color: #ffd66b !important; font-size: 2rem; font-weight: 800; }
    .ticker-wrap {
        width: 100%; overflow: hidden; white-space: nowrap;
        border: 1px solid #a875ff; border-radius: 12px;
        background: rgba(125, 55, 210, 0.25); padding: 10px 0; margin: 8px 0 18px;
    }
    .ticker {
        display: inline-block; padding-left: 100%;
        animation: ticker 13s linear infinite;
        color: #f3d8ff; font-size: 1.15rem; font-weight: 800;
    }
    @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06); padding: 12px;
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.10);
    }
    </style>
    <div class="ticker-wrap"><div class="ticker">💜 TE AMO MAMI 💜 &nbsp;&nbsp;&nbsp; 💜 TE AMO MAMI 💜 &nbsp;&nbsp;&nbsp; 💜 TE AMO MAMI 💜</div></div>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Kalshi Signal Bot")
st.caption(
    "Análisis técnico informativo de BTC/USD y EUR/USD. "
    "No ejecuta operaciones ni garantiza resultados."
)

def safe_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan

@st.cache_data(ttl=8, show_spinner=False)
def get_btc_candles():
    url = f"{COINBASE_API}/products/BTC-USD/candles"
    params = {"granularity": CANDLE_SECONDS}
    response = SESSION.get(url, params=params, timeout=12)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError("Coinbase no devolvió velas.")
    df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for col in ["low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    return df.tail(CANDLE_LIMIT).reset_index(drop=True)

@st.cache_data(ttl=15, show_spinner=False)
def get_eurusd_candles():
    params = {"range": "1d", "interval": "1m", "includePrePost": "true"}
    response = SESSION.get(YAHOO_API, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(str(chart["error"]))
    result = (chart.get("result") or [None])[0]
    if not result:
        raise ValueError("Yahoo Finance no devolvió datos EUR/USD.")
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps:
        raise ValueError("No hay timestamps para EUR/USD.")
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", [0] * len(timestamps)),
    })
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.tail(CANDLE_LIMIT).reset_index(drop=True)

def remove_unclosed_candles(df):
    if df is None or df.empty:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC")
    # Una vela de 1 minuto se considera cerrada tras terminar y pasar 2 segundos.
    cutoff = now - pd.Timedelta(seconds=2)
    out = df[df["timestamp"] + pd.Timedelta(seconds=CANDLE_SECONDS) <= cutoff].copy()
    return out.reset_index(drop=True)

def calculate_indicators(df):
    out = df.copy()
    close = out["close"].astype(float)
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi"] = out["rsi"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["momentum3"] = close.pct_change(3) * 100

    prev_close = close.shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14, min_periods=14).mean()
    out["previous_high"] = out["high"].shift(1).rolling(10, min_periods=10).max()
    out["previous_low"] = out["low"].shift(1).rolling(10, min_periods=10).min()
    return out

def analyze_direction(df):
    """Devuelve BUY UP, BUY DOWN o WAIT a partir de velas cerradas."""
    if df is None or len(df) < MIN_CANDLES:
        return "WAIT", "No hay suficientes velas cerradas", 0, 0

    last = df.iloc[-1]
    required = ["ema9", "ema21", "ema50", "macd", "macd_signal",
                "macd_hist", "momentum3", "rsi", "atr",
                "previous_high", "previous_low"]
    if any(pd.isna(last.get(c, np.nan)) for c in required):
        return "WAIT", "Indicadores todavía no disponibles", 0, 0

    up = 0
    down = 0

    if last["ema9"] > last["ema21"] > last["ema50"]:
        up += 2
    elif last["ema9"] < last["ema21"] < last["ema50"]:
        down += 2

    if last["close"] > last["ema21"]:
        up += 1
    elif last["close"] < last["ema21"]:
        down += 1

    if last["macd"] > last["macd_signal"]:
        up += 1
    elif last["macd"] < last["macd_signal"]:
        down += 1

    if last["macd_hist"] > df.iloc[-2]["macd_hist"]:
        up += 1
    elif last["macd_hist"] < df.iloc[-2]["macd_hist"]:
        down += 1

    if last["momentum3"] > 0:
        up += 1
    elif last["momentum3"] < 0:
        down += 1

    if last["rsi"] >= 75:
        up -= 1
    if last["rsi"] <= 25:
        down -= 1

    if last["close"] > last["previous_high"]:
        up += 1
    elif last["close"] < last["previous_low"]:
        down += 1

    recent_range = df["high"].tail(10).max() - df["low"].tail(10).min()
    if pd.notna(last["atr"]) and recent_range < last["atr"] * 1.5:
        return "WAIT", "Rango reciente estrecho; falta movimiento", up, down

    if up >= 5 and up - down >= 2:
        return "BUY UP", "Confirmaciones técnicas alcistas", up, down
    if down >= 5 and down - up >= 2:
        return "BUY DOWN", "Confirmaciones técnicas bajistas", up, down
    return "WAIT", "Señal sin suficientes confirmaciones", up, down

def get_15m_cycle():
    now = datetime.now(timezone.utc)
    elapsed = (now.minute % 15) * 60 + now.second
    remaining = 15 * 60 - elapsed
    cycle_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    cycle_end = cycle_start + timedelta(minutes=15)
    return now, cycle_start, cycle_end, remaining

def get_kalshi_market(series_ticker):
    """Consulta un mercado abierto de Kalshi; falla de forma segura si el endpoint cambia."""
    try:
        response = SESSION.get(
            f"{KALSHI_API}/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 100},
            timeout=8,
        )
        if response.status_code != 200:
            return None
        body = response.json()
        markets = body.get("markets", []) if isinstance(body, dict) else []
        now = datetime.now(timezone.utc)
        candidates = []
        for market in markets:
            expiry = (
                market.get("close_time")
                or market.get("expected_expiration_time")
                or market.get("expiration_time")
            )
            if not expiry:
                continue
            try:
                expiry_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if expiry_dt > now:
                candidates.append((expiry_dt, market))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None
    except (requests.RequestException, ValueError, TypeError):
        return None

def get_yes_probability(market):
    if not market:
        return None
    for key in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"):
        value = safe_float(market.get(key))
        if np.isfinite(value):
            return round(value * 100, 2)
    for key in ("yes_bid", "yes_ask", "last_price"):
        value = safe_float(market.get(key))
        if np.isfinite(value):
            return round(value, 2)
    return None

def build_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=asset
    ))
    for column, name in [("ema9", "EMA 9"), ("ema21", "EMA 21"), ("ema50", "EMA 50")]:
        if column in df:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[column], mode="lines", name=name))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,8,35,0.65)",
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# Streamlit 1.37+ supports scheduled fragments; no external autorefresh component needed.
@st.fragment(run_every=REFRESH_SECONDS)
def dashboard():
    asset = st.selectbox("Activo", list(ASSETS.keys()), key="selected_asset")
    source = ASSETS[asset]["source"]

    try:
        raw = get_btc_candles() if asset == "BTC/USD" else get_eurusd_candles()
        closed = remove_unclosed_candles(raw)
        if len(closed) < MIN_CANDLES:
            st.warning(f"Solo hay {len(closed)} velas cerradas. Se necesitan al menos {MIN_CANDLES}.")
            st.caption(f"Fuente consultada: {source}")
            return
        data = calculate_indicators(closed)
        signal, reason, up_score, down_score = analyze_direction(data)
        last = data.iloc[-1]
        previous = data.iloc[-2]
        price = float(last["close"])
        change = price - float(previous["close"])
        change_pct = (change / float(previous["close"]) * 100) if previous["close"] else 0.0

        now, cycle_start, cycle_end, remaining = get_15m_cycle()
        mins, secs = divmod(remaining, 60)

        col1, col2, col3 = st.columns(3)
        col1.metric("Precio de referencia", f"{price:,.5f}" if asset == "EUR/USD" else f"${price:,.2f}",
                    f"{change:+.5f} ({change_pct:+.3f}%)")
        col2.metric("Ciclo de 15 minutos", f"{mins:02d}:{secs:02d}", "hasta el siguiente cuarto de hora")
        col3.metric("Última vela cerrada (UTC)", last["timestamp"].strftime("%H:%M:%S"))

        css_class = "signal-wait"
        if signal == "BUY UP":
            css_class = "signal-up"
        elif signal == "BUY DOWN":
            css_class = "signal-down"
        st.markdown(
            f'<div class="signal-card"><div>SEÑAL TÉCNICA</div>'
            f'<div class="{css_class}">{signal}</div><div>{reason}</div></div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Puntuación alcista", f"{up_score}")
        c2.metric("Puntuación bajista", f"{down_score}")
        c3.metric("RSI (14)", f"{last['rsi']:.1f}")

        st.caption(
            f"Fuente: {source}. Actualización aproximada cada {REFRESH_SECONDS} s. "
            "La señal usa velas cerradas de 1 minuto, por eso puede no cambiar en cada actualización."
        )
        st.plotly_chart(build_chart(data.tail(100), asset), use_container_width=True)

        # Kalshi market lookup is optional; does not determine the technical signal.
        series = "KXBTC15M" if asset == "BTC/USD" else "KXEURUSD15M"
        with st.expander("Información de mercado Kalshi (si está disponible)"):
            market = get_kalshi_market(series)
            if market:
                st.write("Ticker:", market.get("ticker", "No disponible"))
                st.write("Título:", market.get("title", "No disponible"))
                prob = get_yes_probability(market)
                st.write("Precio/probabilidad YES reportada:", f"{prob:.2f}" if prob is not None else "No disponible")
            else:
                st.info("No se encontró un mercado abierto o el endpoint no respondió. Esto no afecta el análisis técnico.")
        st.caption(f"Ciclo UTC: {cycle_start.strftime('%H:%M')}–{cycle_end.strftime('%H:%M')}.")
    except requests.Timeout:
        st.error("La fuente de datos tardó demasiado en responder. Se volverá a intentar en la próxima actualización.")
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la fuente de datos: {exc}")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        st.error(f"No se pudieron procesar los datos: {exc}")
    except Exception as exc:
        st.error(f"Error inesperado: {type(exc).__name__}: {exc}")

dashboard()

st.divider()
st.caption(
    "Aviso: BUY UP / BUY DOWN son etiquetas de un modelo técnico sencillo, no predicciones seguras ni recomendaciones financieras. "
    "EUR/USD de Yahoo Finance puede tener retraso; Coinbase proporciona datos del mercado BTC-USD de Coinbase, no de todo el mercado."
)
