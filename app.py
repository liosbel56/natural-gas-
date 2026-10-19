
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Kalshi Early Signal Bot",
    page_icon="📈",
    layout="wide"
)

API = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "KalshiEarlySignalBot/1.0"}

SERIES = {
    "BTC/USD": "KXBTC15M",
    "EUR/USD": "KXEURUSD15M",
}

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #10091f, #1b1030);
    color: #f5efff;
}
h1,h2,h3,p,label {
    color: #f5efff !important;
}
div[data-testid="stMetric"] {
    background: #291744;
    border: 1px solid #7545a8;
    padding: 12px;
    border-radius: 12px;
}
.signal {
    background: #291744;
    border: 1px solid #7545a8;
    border-radius: 14px;
    text-align: center;
    padding: 18px;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Kalshi Early Signal Bot")
st.caption("BTC/USD · EUR/USD · Ventanas de 15 minutos · Sin compras automáticas")

st_autorefresh(interval=5000, key="refresh")

market_name = st.selectbox("Selecciona el mercado", list(SERIES.keys()))
series_ticker = SERIES[market_name]

with st.expander("⚙️ Ajustes de las señales"):
    min_score = st.slider("Confluencia mínima", 3, 7, 5)
    min_adx = st.slider("ADX mínimo", 10, 35, 18)
    max_entry_price = st.slider(
        "Precio máximo de contrato que quieres considerar ($)",
        0.10, 0.95, 0.60, 0.01
    )

# =====================================================
# UTILIDADES
# =====================================================
def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def format_time_remaining(close_time):
    close_dt = parse_time(close_time)
    if not close_dt:
        return "No disponible"

    seconds = max(
        0, int((close_dt - datetime.now(timezone.utc)).total_seconds())
    )
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)

    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# =====================================================
# CONTRATO REAL DE KALSHI
# =====================================================
@st.cache_data(ttl=5, show_spinner=False)
def fetch_kalshi_series(series):
    url = f"{API}/markets"
    params = {"series_ticker": series, "limit": 100}

    response = requests.get(
        url, params=params, headers=HEADERS, timeout=12
    )
    response.raise_for_status()
    payload = response.json()

    markets = payload.get("markets", [])
    now = datetime.now(timezone.utc)

    active = []

    for item in markets:
        status = str(item.get("status", "")).lower()
        start = parse_time(item.get("open_time"))
        close = parse_time(item.get("close_time"))

        # Los nombres de estado pueden variar entre etapas del mercado.
        if (
            status in ("active", "open")
            and start is not None
            and close is not None
            and start <= now < close
        ):
            active.append(item)

    active.sort(key=lambda x: x.get("close_time", ""))
    return active


def dollars(value):
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "N/D"


# =====================================================
# DATOS DEL ACTIVO
# =====================================================
@st.cache_data(ttl=10, show_spinner=False)
def fetch_btc():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    response = requests.get(
        url,
        params={"granularity": 60},
        headers=HEADERS,
        timeout=12
    )
    response.raise_for_status()

    rows = response.json()
    if not rows:
        raise ValueError("Coinbase no devolvió velas de BTC.")

    df = pd.DataFrame(
        rows,
        columns=["time", "low", "high", "open", "close", "volume"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .tail(350)
        .reset_index(drop=True)
    )


@st.cache_data(ttl=20, show_spinner=False)
def fetch_eurusd():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"
    response = requests.get(
        url,
        params={"range": "1d", "interval": "1m"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12
    )
    response.raise_for_status()

    results = response.json()["chart"]["result"]
    if not results:
        raise ValueError("Yahoo Finance no devolvió EUR/USD.")

    result = results[0]
    quote = result["indicators"]["quote"][0]
    timestamps = result.get("timestamp", [])

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote.get("volume", [None] * len(timestamps))
    })

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["open", "high", "low", "close"])
        .drop_duplicates("time")
        .sort_values("time")
        .tail(350)
        .reset_index(drop=True)
    )


# =====================================================
# INDICADORES TÉCNICOS
# =====================================================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def indicators(df):
    df = df.copy()
    close = df["close"]

    for period in [9, 21, 50, 200]:
        df[f"ema{period}"] = close.ewm(
            span=period, adjust=False
        ).mean()

    df["rsi"] = calculate_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    lowest = df["low"].rolling(14).min()
    highest = df["high"].rolling(14).max()
    spread = (highest - lowest).replace(0, np.nan)

    df["stoch_k"] = (
        100 * (close - lowest) / spread
    ).fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(50)

    previous_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=df.index
    )

    plus_di = 100 * plus_dm.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean() / atr.replace(0, np.nan)

    minus_di = 100 * minus_dm.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    df["atr"] = atr
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    df["momentum3"] = close.pct_change(3) * 100
    df["momentum5"] = close.pct_change(5) * 100

    return df.replace([np.inf, -np.inf], np.nan)


# =====================================================
# CONFIRMACIÓN DE 5 MINUTOS
# =====================================================
def trend_5m(df):
    indexed = df.set_index("time").sort_index()

    bars = indexed.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])

    if len(bars) < 30:
        return "UNKNOWN"

    bars = indicators(bars.reset_index())
    last = bars.iloc[-1]

    if last["ema9"] > last["ema21"] and last["close"] > last["ema50"]:
        return "UP"
    if last["ema9"] < last["ema21"] and last["close"] < last["ema50"]:
        return "DOWN"

    return "MIXED"


# =====================================================
# SEÑAL TEMPRANA CON FILTROS
# =====================================================
def signal_engine(df, threshold, adx_min):
    ready = df.dropna(subset=[
        "ema9", "ema21", "ema50", "ema200", "rsi",
        "macd", "macd_signal", "macd_hist",
        "adx", "plus_di", "minus_di", "stoch_k",
        "stoch_d", "momentum3", "momentum5"
    ]).reset_index(drop=True)

    if len(ready) < 205:
        return {
            "signal": "WAIT",
            "score": 0,
            "trend": "UNKNOWN",
            "reasons": ["Esperando suficientes datos para confirmar indicadores."]
        }

    last = ready.iloc[-1]
    prev = ready.iloc[-2]
    trend = trend_5m(df)

    score = 0
    reasons = []

    # Tendencia
    for condition, points, description in [
        (last["ema9"] > last["ema21"], 1, "EMA 9 > EMA 21"),
        (last["close"] > last["ema50"], 1, "Precio > EMA 50"),
        (last["close"] > last["ema200"], 1, "Precio > EMA 200"),
    ]:
        if condition:
            score += points
            reasons.append(description)

    if last["ema9"] < last["ema21"]:
        score -= 1
        reasons.append("EMA 9 < EMA 21")
    if last["close"] < last["ema50"]:
        score -= 1
        reasons.append("Precio < EMA 50")
    if last["close"] < last["ema200"]:
        score -= 1
        reasons.append("Precio < EMA 200")

    # MACD
    if last["macd"] > last["macd_signal"] and last["macd_hist"] > prev["macd_hist"]:
        score += 1
        reasons.append("MACD alcista y mejorando")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < prev["macd_hist"]:
        score -= 1
        reasons.append("MACD bajista y empeorando")

    # RSI
    if 50 < last["rsi"] < 70:
        score += 1
        reasons.append("RSI alcista sin sobrecompra extrema")
    elif 30 < last["rsi"] < 50:
        score -= 1
        reasons.append("RSI bajista")

    # Estocástico
    if last["stoch_k"] > last["stoch_d"] and last["stoch_k"] < 80:
        score += 1
        reasons.append("Estocástico alcista")
    elif last["stoch_k"] < last["stoch_d"] and last["stoch_k"] > 20:
        score -= 1
        reasons.append("Estocástico bajista")

    # Momentum
    if last["momentum3"] > 0 and last["momentum5"] > 0:
        score += 1
        reasons.append("Momentum positivo")
    elif last["momentum3"] < 0 and last["momentum5"] < 0:
        score -= 1
        reasons.append("Momentum negativo")

    # Filtros estrictos
    long_ok = (
        score >= threshold
        and trend == "UP"
        and last["adx"] >= adx_min
        and last["plus_di"] > last["minus_di"]
        and 50 < last["rsi"] < 70
    )

    short_ok = (
        score <= -threshold
        and trend == "DOWN"
        and last["adx"] >= adx_min
        and last["minus_di"] > last["plus_di"]
        and 30 < last["rsi"] < 50
    )

    if long_ok:
        signal = "BUY UP"
    elif short_ok:
        signal = "BUY DOWN"
    else:
        signal = "WAIT"

    reasons.append(f"Tendencia 5 min: {trend}")
    reasons.append(f"ADX: {last['adx']:.1f}")
    reasons.append(f"RSI: {last['rsi']:.1f}")

    return {
        "signal": signal,
        "score": int(score),
        "trend": trend,
        "reasons": reasons,
        "last": last
    }


# =====================================================
# PANTALLA: CONTRATO KALSHI
# =====================================================
st.subheader(f"Kalshi · {market_name}")
st.caption(f"Serie consultada: {series_ticker}")

try:
    contracts = fetch_kalshi_series(series_ticker)

    if contracts:
        labels = [
            f"{m.get('ticker', 'Sin ticker')} · cierre "
            f"{m.get('close_time', 'N/D')}"
            for m in contracts
        ]

        selected_index = st.selectbox(
            "Contrato activo detectado",
            range(len(contracts)),
            format_func=lambda i: labels[i]
        )
        contract = contracts[selected_index]

        col1, col2, col3 = st.columns(3)
        col1.metric("Estado", contract.get("status", "N/D"))
        col2.metric("Tiempo hasta cierre", format_time_remaining(
            contract.get("close_time")
        ))
        col3.metric("YES ask", dollars(contract.get("yes_ask_dollars")))

        d1, d2, d3 = st.columns(3)
        d1.metric("NO ask", dollars(contract.get("no_ask_dollars")))
        d2.metric("YES bid", dollars(contract.get("yes_bid_dollars")))
        d3.metric("NO bid", dollars(contract.get("no_bid_dollars")))

        st.write("**Ticker del contrato:**", contract.get("ticker", "N/D"))
        st.write("**Pregunta:**", contract.get("title", "No disponible"))
        st.write("**YES significa:**", contract.get("yes_sub_title", "Ver reglas del contrato"))
        st.write("**Reglas:**", contract.get("rules_primary", "Consultar la página de Kalshi"))

        st.caption(
            "El contrato y sus precios proceden de la API pública. "
            "Verifica las reglas antes de interpretar YES como UP o NO como DOWN."
        )
    else:
        contract = None
        st.warning(
            "La API no devolvió un contrato abierto para esta serie en este momento. "
            "No se recomienda operar hasta verificar el mercado en Kalshi."
        )

except requests.RequestException as e:
    contract = None
    st.error(f"No se pudo consultar Kalshi: {e}")

# =====================================================
# ANÁLISIS DEL ACTIVO
# =====================================================
st.divider()
st.subheader(f"Indicadores técnicos · {market_name}")

try:
    with st.spinner("Descargando datos y calculando indicadores..."):
        raw = fetch_btc() if market_name == "BTC/USD" else fetch_eurusd()
        df = indicators(raw)

    result = signal_engine(df, min_score, min_adx)
    last = result.get("last", df.iloc[-1])

    a, b, c, d = st.columns(4)
    a.metric("Precio del activo", f"{last['close']:,.5f}")
    b.metric("RSI", f"{last['rsi']:.1f}" if "rsi" in last else "N/D")
    b.metric("ADX", f"{last['adx']:.1f}" if "adx" in last else "N/D")
    c.metric("EMA 9", f"{last['ema9']:,.5f}")
    d.metric("Tendencia 5 min", result["trend"])

    st.markdown(
        f'<div class="signal">{result["signal"]}</div>',
        unsafe_allow_html=True
    )
    st.write(f"**Puntuación:** {result['score']}")
    st.write("**Confirmaciones y filtros:**")
    for reason in result["reasons"]:
        st.write(f"- {reason}")

    # Gráfico de velas
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Precio"
    ))

    for col, label in [
        ("ema9", "EMA 9"),
        ("ema21", "EMA 21"),
        ("ema50", "EMA 50"),
        ("ema200", "EMA 200")
    ]:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[col], name=label, mode="lines"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#10091f",
        plot_bgcolor="#10091f",
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=df["time"], y=df["rsi"], name="RSI 14", mode="lines"
    ))
    rsi_fig.add_hline(y=70, line_dash="dash")
    rsi_fig.add_hline(y=30, line_dash="dash")
    rsi_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#10091f",
        plot_bgcolor="#10091f",
        height=220,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(rsi_fig, use_container_width=True)

    # =================================================
    # COMPROBACIÓN DEL PRECIO DEL CONTRATO
    # =================================================
    st.divider()
    st.subheader("Revisión de la entrada")

    if not contract:
        st.warning(
            "No hay contrato activo verificado; la señal técnica no debe "
            "interpretarse como una orden de entrada en Kalshi."
        )
    else:
        side = "YES" if result["signal"] == "BUY UP" else (
            "NO" if result["signal"] == "BUY DOWN" else None
        )

        st.write(
            "**Lado candidato:** "
            + (side if side else "Ninguno — WAIT")
        )

        if side == "YES":
            candidate_price = contract.get("yes_ask_dollars")
        elif side == "NO":
            candidate_price = contract.get("no_ask_dollars")
        else:
            candidate_price = None

        if candidate_price is not None and side:
            try:
                price = float(candidate_price)
                st.metric("Precio de compra estimado", f"${price:.2f}")

                if price > max_entry_price:
                    st.warning(
                        "El precio supera tu límite. No se considera una "
                        "entrada aceptable según tu filtro."
                    )
                else:
                    st.info(
                        "El precio está dentro de tu límite, pero eso no "
                        "demuestra que exista ventaja estadística."
                    )
            except (TypeError, ValueError):
                st.warning("Kalshi no publicó un precio de compra válido.")

    st.caption(
        "BUY UP/DOWN es una estimación de dirección técnica. No es una "
        "probabilidad validada de ganar el contrato. No se envían órdenes."
    )

except requests.RequestException as e:
    st.error(f"Error descargando datos del activo: {e}")
except (KeyError, ValueError, IndexError, TypeError) as e:
    st.error(f"Datos insuficientes o formato inesperado: {e}")
except Exception as e:
    st.error(f"Error inesperado: {e}")
