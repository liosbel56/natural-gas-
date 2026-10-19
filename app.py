
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIGURACION
# =====================================================
st.set_page_config(
    page_title="Kalshi Signal Bot",
    page_icon="📈",
    layout="wide",
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.exchange.coinbase.com"
BIQUOTE_API = "https://biquote.io/api"

REFRESH_MS = 5000
MAX_CANDLE_AGE_SECONDS = 180
MIN_CANDLES = 60
MIN_SECONDS_TO_CLOSE = 60

st_autorefresh(interval=REFRESH_MS, key="refresh")

st.title("📈 KALSHI SIGNAL BOT")
st.caption(
    "Bitcoin + EUR/USD | Velas de 1 minuto | "
    "Filtros conservadores | Solo señales informativas"
)
st.warning(
    "Modo informativo. No ejecuta compras reales. "
    "Las señales no garantizan beneficios ni evitan todas las pérdidas."
)

# =====================================================
# PETICIONES Y UTILIDADES
# =====================================================
@st.cache_data(ttl=3, show_spinner=False)
def fetch_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        timeout=12,
        headers={"User-Agent": "KalshiSignalBot/2.0"},
    )
    response.raise_for_status()
    return response.json()


def num(value):
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def price_cents(market, *fields):
    for field in fields:
        value = num(market.get(field))
        if np.isfinite(value):
            if field.endswith("_dollars"):
                return value * 100
            return value
    return np.nan


# =====================================================
# KALSHI: CONTRATO ABIERTO MAS PROXIMO A CERRAR
# =====================================================
@st.cache_data(ttl=3, show_spinner=False)
def get_market(series):
    data = fetch_json(
        f"{KALSHI_API}/markets",
        params={
            "series_ticker": series,
            "status": "open",
            "limit": 100,
        },
    )

    now = datetime.now(timezone.utc).timestamp()
    candidates = []

    for market in data.get("markets", []):
        close_text = (
            market.get("close_time")
            or market.get("expiration_time")
        )

        if not close_text:
            continue

        try:
            close_dt = datetime.fromisoformat(
                close_text.replace("Z", "+00:00")
            )
            close_ts = close_dt.timestamp()
        except (TypeError, ValueError):
            continue

        if close_ts > now:
            item = dict(market)
            item["_close_ts"] = close_ts
            candidates.append(item)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["_close_ts"])
    return candidates[0]


def get_yes_probability(market):
    bid = price_cents(market, "yes_bid_dollars", "yes_bid")
    ask = price_cents(market, "yes_ask_dollars", "yes_ask")
    last = price_cents(market, "last_price_dollars", "last_price")

    if (
        np.isfinite(bid)
        and np.isfinite(ask)
        and ask >= bid
    ):
        return (bid + ask) / 2

    if np.isfinite(last):
        return last

    if np.isfinite(bid):
        return bid

    return np.nan


def get_strike(market):
    floor = num(market.get("floor_strike"))
    cap = num(market.get("cap_strike"))
    strike = num(market.get("strike_price"))

    if np.isfinite(floor) and np.isfinite(cap):
        return f"Entre {floor:g} y {cap:g}"

    if np.isfinite(floor):
        return f"Desde {floor:g}"

    if np.isfinite(cap):
        return f"Hasta {cap:g}"

    if np.isfinite(strike):
        return f"{strike:g}"

    return "No confirmado por el API"


# =====================================================
# BITCOIN: COINBASE
# =====================================================
@st.cache_data(ttl=5, show_spinner=False)
def get_btc_candles():
    data = fetch_json(
        f"{COINBASE_API}/products/BTC-USD/candles",
        params={"granularity": 60},
    )

    if not isinstance(data, list) or len(data) < MIN_CANDLES:
        raise RuntimeError(
            "Coinbase devolvió pocas velas."
        )

    df = pd.DataFrame(
        data,
        columns=[
            "time", "low", "high", "open", "close", "volume"
        ],
    )

    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("time")
        .dropna(subset=["time", "open", "high", "low", "close"])
        .drop_duplicates(subset=["time"], keep="last")
    )

    # Excluir la vela del minuto actual, aun si el proveedor
    # ya la incluye en la respuesta.
    current_minute = pd.Timestamp.now(tz="UTC").floor("min")
    cutoff = current_minute - pd.Timedelta(minutes=1)
    df = df[df["time"] <= cutoff]

    valid = (
        (df["open"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df[valid].tail(120).reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        raise RuntimeError(
            "No hay suficientes velas cerradas de Bitcoin."
        )

    return df


# =====================================================
# EUR/USD: BIQUOTE
# =====================================================
@st.cache_data(ttl=5, show_spinner=False)
def get_eurusd_candles():
    data = fetch_json(
        f"{BIQUOTE_API}/EURUSD/ohlc",
        params={"interval": "1m", "limit": 120},
    )

    bars = data.get("bars") if isinstance(data, dict) else None

    if not bars:
        raise RuntimeError(
            "El proveedor EUR/USD no devolvió velas."
        )

    df = pd.DataFrame(bars)

    required = ["openTime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError(
            "Faltan campos en EUR/USD: " + ", ".join(missing)
        )

    df["time"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("time")
        .dropna(subset=["time", "open", "high", "low", "close"])
        .drop_duplicates(subset=["time"], keep="last")
    )

    # Si el proveedor identifica expresamente las velas abiertas,
    # descartarlas. Si no ofrece este campo, se utiliza el tiempo.
    if "isOpen" in df.columns:
        is_open = (
            df["isOpen"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[~is_open]

    # Una vela de un minuto debe haber terminado.
    current_minute = pd.Timestamp.now(tz="UTC").floor("min")
    cutoff = current_minute - pd.Timedelta(minutes=1)
    df = df[df["time"] <= cutoff]

    valid = (
        (df["open"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df[valid].tail(120).reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        raise RuntimeError(
            "No hay suficientes velas cerradas de EUR/USD."
        )

    return df


# =====================================================
# INDICADORES TECNICOS
# =====================================================
def add_indicators(df):
    df = df.copy()
    close = df["close"].astype(float)

    df["ema9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    delta = close.diff()

    gains = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    losses = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = gains / losses.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Casos límite del RSI.
    df.loc[
        (losses == 0) & (gains > 0), "rsi"
    ] = 100

    df.loc[
        (gains == 0) & (losses > 0), "rsi"
    ] = 0

    df.loc[
        (gains == 0) & (losses == 0), "rsi"
    ] = 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["momentum3"] = close.pct_change(3) * 100
    df["momentum1"] = close.pct_change() * 100

    return df


# =====================================================
# SENAL CONSERVADORA
# =====================================================
def make_signal(df, seconds_left=None):
    if df is None or len(df) < MIN_CANDLES:
        return "ESPERAR", 0, [
            "Historial insuficiente para calcular indicadores."
        ]

    row = df.iloc[-1]

    if len(df) < 2:
        return "ESPERAR", 0, [
            "No hay suficientes velas para analizar."
        ]

    last_time = pd.to_datetime(row["time"], utc=True)
    age = (
        pd.Timestamp.now(tz="UTC") - last_time
    ).total_seconds()

    if age < 0 or age > MAX_CANDLE_AGE_SECONDS:
        return "ESPERAR", 0, [
            f"Datos desactualizados: {age:.0f} segundos."
        ]

    required = [
        "ema9", "ema21", "ema50",
        "macd", "macd_signal", "momentum3", "rsi",
    ]

    if any(
        not np.isfinite(num(row[col]))
        for col in required
    ):
        return "ESPERAR", 0, [
            "Los indicadores no están listos o contienen valores inválidos."
        ]

    if seconds_left is not None and seconds_left < MIN_SECONDS_TO_CLOSE:
        return "ESPERAR", 0, [
            "Demasiado cerca del cierre del contrato."
        ]

    bullish_trend = (
        row["ema9"] > row["ema21"] > row["ema50"]
    )

    bearish_trend = (
        row["ema9"] < row["ema21"] < row["ema50"]
    )

    bullish_macd = row["macd"] > row["macd_signal"]
    bearish_macd = row["macd"] < row["macd_signal"]

    bullish_momentum = row["momentum3"] > 0
    bearish_momentum = row["momentum3"] < 0

    rsi = float(row["rsi"])

    # Zonas de RSI conservadoras.
    bullish_rsi = 50 <= rsi <= 65
    bearish_rsi = 35 <= rsi <= 50

    score = (
        int(bullish_trend) - int(bearish_trend)
        + int(bullish_macd) - int(bearish_macd)
        + int(bullish_momentum) - int(bearish_momentum)
    )

    reasons = [
        "Tendencia: " + (
            "alcista" if bullish_trend
            else "bajista" if bearish_trend
            else "sin alineación"
        ),
        "MACD: " + (
            "alcista" if bullish_macd
            else "bajista" if bearish_macd
            else "neutral"
        ),
        f"Momentum de 3 velas: {row['momentum3']:.5f}%",
        f"RSI: {rsi:.1f}",
        f"Antigüedad de la vela: {age:.0f} s",
    ]

    if (
        bullish_trend
        and bullish_macd
        and bullish_momentum
        and bullish_rsi
    ):
        return "POSIBLE SUBIDA", score, reasons + [
            "Filtros alcistas coinciden. No garantiza beneficio."
        ]

    if (
        bearish_trend
        and bearish_macd
        and bearish_momentum
        and bearish_rsi
    ):
        return "POSIBLE CAÍDA", score, reasons + [
            "Filtros bajistas coinciden. No garantiza beneficio."
        ]

    return "ESPERAR", score, reasons + [
        "No hay suficientes confirmaciones."
    ]


# =====================================================
# INTERFAZ DE CADA ACTIVO
# =====================================================
def show_asset(title, series, loader, formatter):
    st.subheader(title)

    market = None

    try:
        market = get_market(series)
    except Exception as exc:
        st.warning(f"Error consultando Kalshi: {exc}")

    try:
        raw_df = loader()
        df = add_indicators(raw_df)
    except Exception as exc:
        st.error(f"Error cargando los datos: {exc}")
        st.info(
            "Comprueba la conexión y que el proveedor siga "
            "ofreciendo el formato de datos esperado."
        )
        return

    seconds_left = None

    if market:
        seconds_left = max(
            0, int(market["_close_ts"] - time.time())
        )

        st.markdown("### Mercado de Kalshi")

        st.write(
            "**Título:**",
            market.get("title", "No disponible")
        )

        st.write(
            "**Ticker:**",
            market.get("ticker", "No disponible")
        )

        st.metric(
            "Cuenta regresiva",
            f"{seconds_left // 60:02d}:{seconds_left % 60:02d}",
        )

        probability = get_yes_probability(market)

        if np.isfinite(probability):
            st.metric(
                "YES / UP implícito",
                f"{probability:.1f}%",
            )
            st.metric(
                "NO / DOWN implícito",
                f"{100 - probability:.1f}%",
            )
        else:
            st.info(
                "No hay cotización suficiente para calcular "
                "la probabilidad implícita."
            )

        columns = st.columns(4)

        quote_fields = [
            ("YES bid", ("yes_bid_dollars", "yes_bid")),
            ("YES ask", ("yes_ask_dollars", "yes_ask")),
            ("NO bid", ("no_bid_dollars", "no_bid")),
            ("NO ask", ("no_ask_dollars", "no_ask")),
        ]

        for column, (label, fields) in zip(
            columns, quote_fields
        ):
            value = price_cents(market, *fields)

            column.metric(
                label,
                f"{value:.1f}¢"
                if np.isfinite(value)
                else "N/D",
            )

        st.write("**Strike detectado:**", get_strike(market))

        st.caption(
            "Confirma las reglas oficiales de liquidación "
            "del contrato antes de tomar decisiones."
        )

    else:
        st.warning(
            f"No se encontró un contrato abierto para {series}. "
            "La señal técnica no confirma que exista un contrato negociable."
        )

    signal, score, reasons = make_signal(
        df, seconds_left
    )

    st.markdown("### Señal conservadora")

    if signal == "POSIBLE SUBIDA":
        st.success("🟢 " + signal)
    elif signal == "POSIBLE CAÍDA":
        st.warning("🔴 " + signal)
    else:
        st.info("⚪ " + signal)

    last = df.iloc[-1]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Precio de referencia",
        formatter(float(last["close"])),
    )

    c2.metric(
        "Puntuación técnica",
        f"{score:+d}",
    )

    c3.metric(
        "RSI",
        f"{last['rsi']:.1f}",
    )

    st.markdown("**Motivos de la señal**")

    for reason in reasons:
        st.write("• " + reason)

    # Aviso de cambios entre actualizaciones.
    state_key = f"previous_signal_{series}"
    old_signal = st.session_state.get(state_key)

    if old_signal is not None and old_signal != signal:
        st.warning(
            f"🔔 CAMBIO DE SEÑAL: {old_signal} → {signal}"
        )

    st.session_state[state_key] = signal

    # Grafica de velas y medias moviles.
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Precio",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["ema9"], name="EMA 9"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["ema21"], name="EMA 21"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["ema50"], name="EMA 50"
        )
    )

    fig.update_layout(
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(l=5, r=5, t=25, b=5),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "BTC usa Coinbase; EUR/USD usa Biquote. Los precios "
        "pueden diferir del índice de liquidación de Kalshi. "
        "Las señales no están validadas mediante backtesting."
    )


# =====================================================
# PESTANAS PRINCIPALES
# =====================================================
tab_btc, tab_eur = st.tabs([
    "₿ BITCOIN",
    "💱 EUR/USD",
])

with tab_btc:
    show_asset(
        "Bitcoin (BTC/USD)",
        "KXBTC15M",
        get_btc_candles,
        lambda price: f"${price:,.2f}",
    )

with tab_eur:
    show_asset(
        "EUR/USD",
        "KXEURUSD15M",
        get_eurusd_candles,
        lambda price: f"{price:.5f}",
    )

st.divider()

st.caption(
    "Modo informativo: no ejecuta órdenes reales. "
    "La estrategia no garantiza rentabilidad. "
    "Prueba con datos históricos antes de considerar "
    "cualquier uso con dinero real."
)
