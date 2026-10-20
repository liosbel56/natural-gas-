
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================
# CONFIGURACION
# =========================================================
st.set_page_config(
    page_title="Kalshi Early Signal Bot",
    page_icon="📈",
    layout="wide",
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.exchange.coinbase.com"
BIQUOTE_API = "https://biquote.io/api"

REFRESH_MS = 5000
CACHE_SECONDS = 5
MAX_MARKET_PRICE_CENTS = 90.0

st_autorefresh(interval=REFRESH_MS, key="main_refresh")

st.title("📈 KALSHI EARLY SIGNAL BOT")
st.caption(
    "Bitcoin y EUR/USD | Velas de 1 minuto | "
    "Deteccion temprana | Contratos de 15 minutos"
)
st.warning(
    "Modo informativo. No ejecuta operaciones. "
    "Las señales son estimaciones, no garantizan resultados."
)

# =========================================================
# UTILIDADES
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "KalshiEarlySignalBot/2.0"})


def fetch_json(url, params=None):
    response = SESSION.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def num(value):
    try:
        if value is None or value == "":
            return np.nan
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def price_cents(market, *fields):
    for field in fields:
        value = num(market.get(field))
        if np.isfinite(value):
            if field.endswith("_dollars"):
                value *= 100
            if 0 <= value <= 100:
                return value
    return np.nan


def parse_timestamp(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.timestamp()
    except (ValueError, TypeError):
        return None


# =========================================================
# KALSHI: BUSQUEDA DE MERCADOS ABIERTOS
# =========================================================
@st.cache_data(ttl=5, show_spinner=False)
def get_market(series):
    data = fetch_json(
        f"{KALSHI_API}/markets",
        params={
            "series_ticker": series,
            "status": "open",
            "limit": 100,
        },
    )

    now = time.time()
    candidates = []

    for market in data.get("markets", []):
        close_ts = parse_timestamp(
            market.get("close_time")
            or market.get("expiration_time")
        )

        if close_ts is None or close_ts <= now:
            continue

        item = dict(market)
        item["_close_ts"] = close_ts
        candidates.append(item)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["_close_ts"])
    return candidates[0]


def get_yes_probability(market):
    bid = price_cents(
        market, "yes_bid_dollars", "yes_bid"
    )
    ask = price_cents(
        market, "yes_ask_dollars", "yes_ask"
    )
    last = price_cents(
        market, "last_price_dollars", "last_price"
    )

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

    if np.isfinite(ask):
        return ask

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

    return "No disponible"


# =========================================================
# BITCOIN: COINBASE
# =========================================================
@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_btc_candles():
    data = fetch_json(
        f"{COINBASE_API}/products/BTC-USD/candles",
        params={"granularity": 60},
    )

    if not isinstance(data, list) or len(data) < 30:
        raise RuntimeError(
            "Coinbase no devolvio suficientes velas."
        )

    df = pd.DataFrame(
        data,
        columns=[
            "time", "low", "high",
            "open", "close", "volume",
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
        .drop_duplicates(subset=["time"])
        .tail(120)
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise RuntimeError("No hay suficientes velas validas.")

    return df


# =========================================================
# EUR/USD: BIQUOTE
# =========================================================
@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_eurusd_candles():
    data = fetch_json(
        f"{BIQUOTE_API}/EURUSD/ohlc",
        params={"interval": "1m", "limit": 150},
    )

    bars = data.get("bars") if isinstance(data, dict) else None

    if not bars:
        raise RuntimeError(
            "Biquote no devolvio velas para EUR/USD."
        )

    df = pd.DataFrame(bars)

    required = ["openTime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError(
            "Faltan columnas de EUR/USD: " + ", ".join(missing)
        )

    df["time"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Se excluye la vela abierta para evitar usar datos parciales.
    if "isOpen" in df.columns:
        df = df[df["isOpen"] != True].copy()

    df = (
        df.sort_values("time")
        .dropna(subset=["time", "open", "high", "low", "close"])
        .drop_duplicates(subset=["time"])
        .tail(120)
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise RuntimeError(
            "Menos de 30 velas cerradas validas de EUR/USD."
        )

    return df


# =========================================================
# INDICADORES TECNICOS
# =========================================================
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

    df["rsi"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["momentum1"] = close.pct_change(1) * 100
    df["momentum3"] = close.pct_change(3) * 100

    # Fuerza relativa del movimiento reciente.
    df["ema_spread_pct"] = (
        (df["ema9"] - df["ema21"]) / close
    ) * 100

    return df


# =========================================================
# DETECCION TEMPRANA
# =========================================================
def make_signal(df, seconds_left=None):
    row = df.iloc[-1]
    previous = df.iloc[-2]

    score = 0
    reasons = []

    # EMA 9 frente a EMA 21.
    if row["ema9"] > row["ema21"]:
        score += 1
        reasons.append("EMA 9 por encima de EMA 21")
    else:
        score -= 1
        reasons.append("EMA 9 por debajo de EMA 21")

    # Tendencia media.
    if row["ema21"] > row["ema50"]:
        score += 1
        reasons.append("Tendencia media alcista")
    else:
        score -= 1
        reasons.append("Tendencia media bajista")

    # MACD.
    if row["macd"] > row["macd_signal"]:
        score += 1
        reasons.append("MACD favorece la subida")
    else:
        score -= 1
        reasons.append("MACD favorece la bajada")

    # Impulso reciente.
    if row["momentum3"] > 0:
        score += 1
        reasons.append("Impulso de 3 velas positivo")
    elif row["momentum3"] < 0:
        score -= 1
        reasons.append("Impulso de 3 velas negativo")
    else:
        reasons.append("Sin impulso claro")

    # Cambios de direccion.
    bullish_cross = (
        previous["ema9"] <= previous["ema21"]
        and row["ema9"] > row["ema21"]
    )

    bearish_cross = (
        previous["ema9"] >= previous["ema21"]
        and row["ema9"] < row["ema21"]
    )

    rsi = float(row["rsi"])
    reasons.append(f"RSI: {rsi:.1f}")

    # Si el mercado esta a punto de cerrar, no presentar
    # una nueva señal de entrada tardia.
    if seconds_left is not None and seconds_left < 30:
        return (
            "ESPERAR CIERRE",
            score,
            reasons + [
                "Quedan menos de 30 segundos del contrato."
            ],
        )

    # Deteccion temprana: basta con dos puntos de acuerdo.
    # Esto NO representa una probabilidad estadistica.
    if score >= 2:
        signal = "POSIBLE SUBIDA"
    elif score <= -2:
        signal = "POSIBLE BAJADA"
    elif score == 1:
        signal = "VIGILAR SUBIDA"
    elif score == -1:
        signal = "VIGILAR BAJADA"
    else:
        signal = "SIN DIRECCION CLARA"

    if bullish_cross:
        reasons.append("Cruce reciente de EMA hacia arriba")
        if score >= 0:
            signal = "POSIBLE SUBIDA"

    if bearish_cross:
        reasons.append("Cruce reciente de EMA hacia abajo")
        if score <= 0:
            signal = "POSIBLE BAJADA"

    # RSI extremo: aviso de posible agotamiento, no
    # una orden automatica de compra o venta.
    if rsi >= 75:
        reasons.append(
            "RSI elevado: posible agotamiento de la subida"
        )
    elif rsi <= 25:
        reasons.append(
            "RSI bajo: posible agotamiento de la bajada"
        )

    return signal, score, reasons


# =========================================================
# INTERFAZ
# =========================================================
def show_asset(title, series, loader, formatter):
    st.subheader(title)

    market = None
    market_error = None

    try:
        market = get_market(series)
    except Exception as exc:
        market_error = str(exc)

    if market_error:
        st.warning(
            "No se pudo consultar Kalshi en esta actualizacion: "
            + market_error
        )

    try:
        df = add_indicators(loader())
    except Exception as exc:
        st.error("Error cargando los datos: " + str(exc))
        st.info(
            "Revisa la conexion, el proveedor de datos y "
            "las dependencias instaladas."
        )
        return

    seconds_left = None

    # Informacion del contrato de Kalshi.
    if market:
        seconds_left = max(
            0, int(market["_close_ts"] - time.time())
        )

        st.markdown("### Mercado de Kalshi")
        st.write(
            "**Titulo:**",
            market.get("title", "No disponible"),
        )
        st.write(
            "**Ticker:**",
            market.get("ticker", "No disponible"),
        )

        st.metric(
            "Tiempo hasta el cierre",
            f"{seconds_left // 60:02d}:{seconds_left % 60:02d}",
        )

        probability = get_yes_probability(market)

        if np.isfinite(probability):
            st.metric(
                "Precio medio YES (centavos)",
                f"{probability:.1f}¢",
            )
            st.caption(
                "Los centavos representan el precio del contrato, "
                "no una probabilidad calibrada del bot."
            )

            if probability >= MAX_MARKET_PRICE_CENTS:
                st.warning(
                    "YES cotiza a 90¢ o mas. El bot seguira "
                    "mostrando el analisis tecnico temprano, "
                    "pero evita perseguir un contrato tan caro."
                )

            if probability <= 100 - MAX_MARKET_PRICE_CENTS:
                st.warning(
                    "YES cotiza a 10¢ o menos. Puede existir "
                    "riesgo elevado y poca liquidez."
                )
        else:
            st.info("Kalshi no ofrece una cotizacion util ahora.")

        quotes = st.columns(4)
        quote_fields = [
            ("YES bid", ("yes_bid_dollars", "yes_bid")),
            ("YES ask", ("yes_ask_dollars", "yes_ask")),
            ("NO bid", ("no_bid_dollars", "no_bid")),
            ("NO ask", ("no_ask_dollars", "no_ask")),
        ]

        for column, (label, fields) in zip(quotes, quote_fields):
            value = price_cents(market, *fields)
            column.metric(
                label,
                f"{value:.1f}¢"
                if np.isfinite(value)
                else "N/D",
            )

        st.write("**Strike indicado por la API:**", get_strike(market))
        st.caption(
            "Comprueba las reglas de liquidacion del contrato. "
            "YES no significa necesariamente que el precio vaya a subir."
        )
    else:
        st.warning(
            f"No se encontro un contrato abierto para {series}. "
            "La señal tecnica se calcula igualmente con el precio "
            "de referencia, pero no confirma que haya un contrato activo."
        )

    # Senal tecnica independiente de la cotizacion de Kalshi.
    signal, score, reasons = make_signal(df, seconds_left)

    st.markdown("### Señal temprana")

    if signal == "POSIBLE SUBIDA":
        st.success("🟢 " + signal)
    elif signal == "POSIBLE BAJADA":
        st.error("🔴 " + signal)
    elif signal.startswith("VIGILAR"):
        st.warning("🟡 " + signal)
    else:
        st.info("⚪ " + signal)

    last = df.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Precio de referencia",
        formatter(float(last["close"])),
    )
    c2.metric("Puntuacion tecnica", f"{score:+d} / 4")
    c3.metric("RSI", f"{last['rsi']:.1f}")

    st.markdown("**Motivos del analisis**")
    for reason in reasons:
        st.write("• " + reason)

    # Avisar en pantalla solamente cuando cambia la senal.
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
        "BTC usa Coinbase y EUR/USD usa Biquote. Sus precios "
        "pueden diferir de la referencia de liquidacion de Kalshi. "
        "Las actualizaciones dependen de la disponibilidad y frescura "
        "de los proveedores."
    )


# =========================================================
# PESTANAS PRINCIPALES
# =========================================================
tab_btc, tab_eur = st.tabs(["₿ BITCOIN", "💱 EUR/USD"])

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
    "El bot no ejecuta operaciones ni envia notificaciones "
    "al telefono. Los avisos de cambio se muestran dentro de "
    "Streamlit. Las señales no estan validadas mediante backtesting "
    "y no representan probabilidades estadisticas."
)
