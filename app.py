
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Kalshi Signal Bot",
    page_icon="📈",
    layout="wide",
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.exchange.coinbase.com"
YAHOO_API = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"

REFRESH_SECONDS = 10
CANDLE_SECONDS = 60
REQUIRED_CONFIRMATIONS = 2
CANDLE_LIMIT = 200

# Verifica estos tickers en tu cuenta/API de Kalshi.
ASSETS = {
    "BTC/USD": {
        "series_ticker": "KXBTC15M",
        "source": "Coinbase",
    },
    "EUR/USD": {
        "series_ticker": "KXEURUSD15M",
        "source": "Yahoo Finance",
    },
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "MarketSignalBot/1.0",
    "Accept": "application/json",
})


# =========================================================
# FUNCIONES GENERALES
# =========================================================

def safe_float(value, default=np.nan):
    """Convierte un valor a float sin interrumpir la aplicación."""
    try:
        if value is None:
            return default
        result = float(value)
        return result if np.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def utc_now():
    return pd.Timestamp.now(tz="UTC")


def format_price(value):
    if not np.isfinite(value):
        return "N/D"
    return f"{value:,.5f}"


def reset_confirmation(asset):
    key = f"confirmation_{asset}"
    st.session_state[key] = {
        "last_candle": None,
        "direction": None,
        "count": 0,
    }


# =========================================================
# DATOS DE BTC/USD: COINBASE
# =========================================================

@st.cache_data(ttl=8, show_spinner=False)
def get_btc_candles():
    url = f"{COINBASE_API}/products/BTC-USD/candles"

    response = SESSION.get(
        url,
        params={"granularity": CANDLE_SECONDS},
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list) or not payload:
        raise ValueError("Coinbase no devolvió velas de BTC/USD.")

    # Coinbase devuelve:
    # [timestamp, low, high, open, close, volume]
    df = pd.DataFrame(
        payload,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    return df.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# DATOS DE EUR/USD: YAHOO FINANCE
# =========================================================

@st.cache_data(ttl=15, show_spinner=False)
def get_eurusd_candles():
    response = SESSION.get(
        YAHOO_API,
        params={
            "range": "1d",
            "interval": "1m",
        },
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    payload = response.json()

    chart = payload.get("chart", {})
    errors = chart.get("error")

    if errors:
        raise ValueError(f"Yahoo Finance: {errors}")

    results = chart.get("result")

    if not results:
        raise ValueError("Yahoo Finance no devolvió datos de EUR/USD.")

    result = results[0]
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})
    quote_list = indicators.get("quote", [])

    if not timestamps or not quote_list:
        raise ValueError("No hay velas disponibles de EUR/USD.")

    quote = quote_list[0]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            timestamps,
            unit="s",
            utc=True,
        ),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get(
            "volume",
            [0] * len(timestamps),
        ),
    })

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    return df.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# FILTRAR VELAS CERRADAS
# =========================================================

def remove_unclosed_candles(df):
    """
    Excluye la vela que todavía se está formando.
    Se deja un margen de 2 segundos después del cierre.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    result = df.copy()
    now = utc_now()
    cutoff = now - pd.Timedelta(seconds=2)

    candle_end = (
        pd.to_datetime(result["timestamp"], utc=True)
        + pd.Timedelta(seconds=CANDLE_SECONDS)
    )

    result = result.loc[candle_end <= cutoff].copy()

    result = result.dropna(
        subset=["open", "high", "low", "close"]
    )

    return (
        result.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


# =========================================================
# INDICADORES TÉCNICOS
# =========================================================

def calculate_indicators(df):
    if df is None or len(df) < 55:
        raise ValueError(
            "Se necesitan al menos 55 velas cerradas para "
            "calcular los indicadores."
        )

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Medias móviles exponenciales
    df["ema9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    df["ema21"] = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    # RSI de 14 periodos
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + relative_strength))

    # Manejar casos de movimiento exclusivamente alcista o bajista.
    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi",
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "rsi",
    ] = 0

    df.loc[
        (avg_gain == 0) & (avg_loss == 0),
        "rsi",
    ] = 50

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Momentum: cambio porcentual de las últimas 3 velas
    df["momentum"] = close.pct_change(3) * 100

    # Rango verdadero medio (ATR)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # Máximos y mínimos previos, sin incluir la vela actual.
    df["previous_high"] = high.shift(1).rolling(10).max()
    df["previous_low"] = low.shift(1).rolling(10).min()

    return df


# =========================================================
# ANÁLISIS DE DIRECCIÓN
# =========================================================

def analyze_direction(df):
    """
    Produce una dirección preliminar a partir de indicadores.
    No representa una garantía ni una probabilidad estadística.
    """
    if df is None or len(df) < 55:
        return "ESPERAR", "No hay suficientes velas."

    last = df.iloc[-1]

    required = [
        "close",
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "momentum",
        "atr",
        "previous_high",
        "previous_low",
    ]

    if any(
        pd.isna(last[col])
        for col in required
    ):
        return "ESPERAR", "Los indicadores aún no están completos."

    previous = df.iloc[-2]

    bullish_score = 0
    bearish_score = 0

    # Tendencia
    if last["ema9"] > last["ema21"] > last["ema50"]:
        bullish_score += 2
    elif last["ema9"] < last["ema21"] < last["ema50"]:
        bearish_score += 2

    # Precio en relación con la EMA21
    if last["close"] > last["ema21"]:
        bullish_score += 1
    elif last["close"] < last["ema21"]:
        bearish_score += 1

    # MACD
    if last["macd"] > last["macd_signal"]:
        bullish_score += 1
    elif last["macd"] < last["macd_signal"]:
        bearish_score += 1

    # Aceleración del histograma MACD
    if last["macd_hist"] > previous["macd_hist"]:
        bullish_score += 1
    elif last["macd_hist"] < previous["macd_hist"]:
        bearish_score += 1

    # Momentum
    if last["momentum"] > 0:
        bullish_score += 1
    elif last["momentum"] < 0:
        bearish_score += 1

    # Evitar comprar una subida extremadamente sobreextendida
    # o vender una caída extremadamente sobreextendida.
    if last["rsi"] >= 75:
        bullish_score -= 1

    if last["rsi"] <= 25:
        bearish_score -= 1

    # Detectar ruptura del rango anterior.
    breakout_up = last["close"] > last["previous_high"]
    breakout_down = last["close"] < last["previous_low"]

    if breakout_up:
        bullish_score += 1

    if breakout_down:
        bearish_score += 1

    # Filtrar rangos de movimiento muy estrecho.
    recent_range = (
        df["high"].tail(10).max()
        - df["low"].tail(10).min()
    )

    if (
        last["atr"] > 0
        and recent_range < last["atr"] * 1.5
    ):
        return "ESPERAR", "Mercado con movimiento limitado."

    # No exigir una ruptura para todas las señales:
    # permite detectar posibles movimientos antes.
    if (
        bullish_score >= 5
        and bullish_score >= bearish_score + 2
        and last["rsi"] < 75
    ):
        return "SUBIDA", (
            f"Indicios alcistas: puntuación {bullish_score} "
            f"frente a {bearish_score}."
        )

    if (
        bearish_score >= 5
        and bearish_score >= bullish_score + 2
        and last["rsi"] > 25
    ):
        return "BAJADA", (
            f"Indicios bajistas: puntuación {bearish_score} "
            f"frente a {bullish_score}."
        )

    return "ESPERAR", (
        f"Señales mixtas: subida {bullish_score}, "
        f"bajada {bearish_score}."
    )


# =========================================================
# CONFIRMACIÓN DE SEÑALES
# =========================================================

def confirm_new_candle(asset, candle_time, raw_direction):
    """
    Cuenta confirmaciones únicamente en velas distintas.
    Si la dirección cambia o queda en ESPERAR, reinicia el conteo.
    """
    key = f"confirmation_{asset}"

    if key not in st.session_state:
        reset_confirmation(asset)

    state = st.session_state[key]
    candle_id = str(candle_time)

    # Una contradicción invalida la señal previa.
    if raw_direction == "ESPERAR":
        state["direction"] = None
        state["count"] = 0
        state["last_candle"] = candle_id
        return "ESPERAR", 0

    if state["last_candle"] == candle_id:
        if state["direction"] != raw_direction:
            state["direction"] = None
            state["count"] = 0
            return "ESPERAR", 0

        if state["count"] >= REQUIRED_CONFIRMATIONS:
            return state["direction"], state["count"]

        return "ESPERAR", state["count"]

    # Solo se llega aquí con una vela nueva.
    state["last_candle"] = candle_id

    if state["direction"] == raw_direction:
        state["count"] += 1
    else:
        state["direction"] = raw_direction
        state["count"] = 1

    if state["count"] >= REQUIRED_CONFIRMATIONS:
        return state["direction"], state["count"]

    return "ESPERAR", state["count"]


# =========================================================
# CONSULTA DE MERCADOS KALSHI
# =========================================================

@st.cache_data(ttl=10, show_spinner=False)
def get_market(series_ticker):
    """
    Busca mercados abiertos de la serie solicitada.
    Devuelve el mercado más próximo a cerrar, si existe.
    """
    url = f"{KALSHI_API}/markets"

    response = SESSION.get(
        url,
        params={
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 100,
        },
        timeout=15,
    )

    response.raise_for_status()

    payload = response.json()
    markets = payload.get("markets", [])

    if not markets:
        return None

    now = utc_now()
    candidates = []

    for market in markets:
        close_text = (
            market.get("close_time")
            or market.get("expected_expiration_time")
            or market.get("expiration_time")
        )

        if not close_text:
            continue

        try:
            close_time = pd.Timestamp(close_text)

            if close_time.tzinfo is None:
                close_time = close_time.tz_localize("UTC")
            else:
                close_time = close_time.tz_convert("UTC")

            if close_time > now:
                candidates.append((close_time, market))

        except (ValueError, TypeError):
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    market = candidates[0][1]
    market["_parsed_close_time"] = candidates[0][0].isoformat()

    return market


def get_yes_probability(market):
    """
    Precio YES expresado en centavos.
    Los campos terminados en _dollars se convierten a centavos.
    """
    if not market:
        return np.nan

    bid_dollars = safe_float(market.get("yes_bid_dollars"))
    ask_dollars = safe_float(market.get("yes_ask_dollars"))
    last_dollars = safe_float(market.get("last_price_dollars"))

    if (
        np.isfinite(bid_dollars)
        and np.isfinite(ask_dollars)
        and 0 <= bid_dollars <= 1
        and 0 <= ask_dollars <= 1
    ):
        return (bid_dollars + ask_dollars) * 50

    if np.isfinite(last_dollars) and 0 <= last_dollars <= 1:
        return last_dollars * 100

    bid = safe_float(market.get("yes_bid"))
    ask = safe_float(market.get("yes_ask"))
    last = safe_float(market.get("last_price"))

    if np.isfinite(bid) and np.isfinite(ask):
        return (bid + ask) / 2

    if np.isfinite(last):
        return last

    if np.isfinite(bid):
        return bid

    if np.isfinite(ask):
        return ask

    return np.nan


# =========================================================
# GRÁFICA
# =========================================================

def create_chart(df, asset):
    chart_df = df.tail(100)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df["timestamp"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name=asset,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["ema9"],
            mode="lines",
            name="EMA 9",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["ema21"],
            mode="lines",
            name="EMA 21",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["ema50"],
            mode="lines",
            name="EMA 50",
        )
    )

    fig.update_layout(
        title=f"{asset} — velas de 1 minuto",
        xaxis_title="Hora UTC",
        yaxis_title="Precio",
        height=520,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


# =========================================================
# MOSTRAR ACTIVO
# =========================================================

def render_asset(asset):
    config = ASSETS[asset]

    st.subheader(asset)

    # Datos del activo
    try:
        if asset == "BTC/USD":
            raw_df = get_btc_candles()
        else:
            raw_df = get_eurusd_candles()

        df = remove_unclosed_candles(raw_df)

        if len(df) < 55:
            st.warning(
                f"Hay solamente {len(df)} velas cerradas. "
                "Se necesitan al menos 55."
            )
            return

        df = calculate_indicators(df)

    except Exception as exc:
        st.error(f"Error al obtener o procesar {asset}: {exc}")
        return

    last = df.iloc[-1]
    previous = df.iloc[-2]

    current_price = safe_float(last["close"])
    raw_direction, reason = analyze_direction(df)

    confirmed_direction, confirmation_count = confirm_new_candle(
        asset,
        last["timestamp"],
        raw_direction,
    )

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Precio de referencia",
            format_price(current_price),
            f"{current_price - previous['close']:+.5f}",
        )

    with col2:
        st.metric(
            "RSI (14)",
            f"{last['rsi']:.1f}",
        )

    with col3:
        st.metric(
            "EMA 9",
            format_price(last["ema9"]),
        )

    with col4:
        st.metric(
            "EMA 21",
            format_price(last["ema21"]),
        )

    st.caption(
        f"Última vela cerrada: "
        f"{last['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    # Señal preliminar
    st.markdown("#### Señal preliminar")

    if raw_direction == "SUBIDA":
        st.info("Posible SUBIDA — señal técnica preliminar.")
    elif raw_direction == "BAJADA":
        st.info("Posible BAJADA — señal técnica preliminar.")
    else:
        st.warning("ESPERAR — no hay dirección suficientemente clara.")

    st.caption(reason)

    # Señal confirmada
    st.markdown("#### Señal confirmada")

    if confirmed_direction == "SUBIDA":
        st.success("SUBIDA CONFIRMADA por la regla de dos velas.")
    elif confirmed_direction == "BAJADA":
        st.error("BAJADA CONFIRMADA por la regla de dos velas.")
    else:
        st.warning(
            f"ESPERAR — confirmaciones consecutivas: "
            f"{confirmation_count}/{REQUIRED_CONFIRMATIONS}."
        )

    st.caption(
        "La confirmación es una regla técnica del bot, "
        "no una garantía de que el precio vaya a subir o bajar."
    )

    # Datos de Kalshi: separados del análisis técnico
    st.markdown("#### Mercado de Kalshi")

    try:
        market = get_market(config["series_ticker"])

        if market is None:
            st.warning(
                "No se encontró un mercado abierto próximo a cerrar "
                f"para la serie {config['series_ticker']}. "
                "Verifica el ticker en Kalshi."
            )
        else:
            probability = get_yes_probability(market)

            market_col1, market_col2 = st.columns(2)

            with market_col1:
                st.metric(
                    "Precio YES",
                    (
                        f"{probability:.1f}¢"
                        if np.isfinite(probability)
                        else "N/D"
                    ),
                )

            with market_col2:
                st.write(
                    "**Ticker del contrato:**",
                    market.get("ticker", "N/D"),
                )

            close_text = market.get("_parsed_close_time")

            if close_text:
                st.caption(f"Cierre del mercado: {close_text}")

            st.caption(
                "El precio YES de Kalshi es independiente "
                "de la señal técnica."
            )

    except Exception as exc:
        st.warning(f"No se pudo consultar Kalshi: {exc}")

    # Gráfica de velas e indicadores
    st.plotly_chart(
        create_chart(df, asset),
        use_container_width=True,
        key=f"chart_{asset.replace('/', '_')}",
    )

    # Tabla de datos técnicos
    with st.expander("Ver indicadores técnicos"):
        display_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "ema9",
            "ema21",
            "ema50",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "momentum",
            "atr",
        ]

        st.dataframe(
            df[display_cols].tail(20).sort_values(
                "timestamp",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# INTERFAZ PRINCIPAL
# =========================================================

st.title("📈 Kalshi Signal Bot")
st.caption(
    "Análisis técnico de BTC/USD y EUR/USD. "
    "Solo genera señales informativas; no ejecuta operaciones."
)

st_autorefresh(
    interval=REFRESH_SECONDS * 1000,
    key="automatic_refresh",
)

st.sidebar.header("Configuración")

st.sidebar.write(
    f"**Actualización:** cada {REFRESH_SECONDS} segundos"
)

st.sidebar.write(
    f"**Confirmación:** {REQUIRED_CONFIRMATIONS} velas cerradas"
)

st.sidebar.write(
    f"**Temporalidad:** {CANDLE_SECONDS} segundo(s) por vela".replace(
        f"{CANDLE_SECONDS} segundo(s)",
        "1 minuto",
    )
)

st.sidebar.warning(
    "Las señales son orientativas. Ningún indicador puede "
    "predecir con certeza el siguiente movimiento."
)

tab_btc, tab_eurusd = st.tabs(["₿ BTC/USD", "💱 EUR/USD"])

with tab_btc:
    render_asset("BTC/USD")

with tab_eurusd:
    render_asset("EUR/USD")

st.divider()

st.caption(
    "Fuente BTC/USD: Coinbase. Fuente EUR/USD: Yahoo Finance. "
    "Los datos, la disponibilidad de contratos y los retrasos "
    "dependen de los servicios externos."
)

st.caption(
    f"Última actualización de la interfaz: "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
