
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Kalshi Signal Bot",
    page_icon="📈",
    layout="wide",
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.exchange.coinbase.com"
BIQUOTE_API = "https://biquote.io/api"

# Actualización visual cada segundo.
st_autorefresh(interval=1000, key="refresh")

st.title("📈 KALSHI SIGNAL BOT")
st.caption("Bitcoin + EUR/USD | Velas de 1 minuto | Contratos de 15 minutos")
st.warning(
    "Señales estimadas, no garantías. El bot no ejecuta compras reales."
)

# =====================================================
# PETICIONES Y UTILIDADES
# =====================================================
def fetch_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        timeout=12,
        headers={"User-Agent": "KalshiSignalBot/1.0"},
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
# KALSHI: CONTRATO ABIERTO MÁS PRÓXIMO A CERRAR
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

    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
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

    if not isinstance(data, list) or len(data) < 30:
        raise RuntimeError("Coinbase no devolvió suficientes velas.")

    df = pd.DataFrame(
        data,
        columns=["time", "low", "high", "open", "close", "volume"],
    )

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.sort_values("time")
        .tail(120)
        .dropna(subset=["close"])
        .reset_index(drop=True)
    )


# =====================================================
# EUR/USD: BIQUOTE, SIN API KEY
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
            "El proveedor EUR/USD no devolvió velas. "
            "Puede estar caído o haber cambiado su API."
        )

    df = pd.DataFrame(bars)

    required = ["openTime", "open", "high", "low", "close"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(
            "Formato inesperado de EUR/USD; faltan: "
            + ", ".join(missing)
        )

    df["time"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.sort_values("time")
        .dropna(subset=["time", "open", "high", "low", "close"])
        .tail(120)
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise RuntimeError("Hay menos de 30 velas válidas de EUR/USD.")

    return df


# =====================================================
# INDICADORES TÉCNICOS
# =====================================================
def add_indicators(df):
    df = df.copy()
    close = df["close"].astype(float)

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()
    losses = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = gains / losses.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

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
# SEÑALES: CONFIRMACIÓN OBLIGATORIA DE LOS 4 INDICADORES
# =====================================================
def make_signal(df, seconds_left=None):
    row = df.iloc[-1]
    previous = df.iloc[-2]

    score = 0
    reasons = []

    # Evitar señales demasiado cerca del cierre.
    if seconds_left is not None and seconds_left < 30:
        return "ESPERAR", 0, [
            "Quedan menos de 30 segundos para el cierre.",
            "Se requiere esperar al siguiente contrato."
        ]

    # Comprobar que los indicadores necesarios sean válidos.
    required = [
        "ema9", "ema21", "ema50",
        "macd", "macd_signal", "momentum3", "rsi"
    ]

    for column in required:
        if column not in df.columns or not np.isfinite(row[column]):
            return "ESPERAR", 0, [
                f"Datos insuficientes o inválidos para {column}."
            ]

    for column in ["ema9", "ema21"]:
        if not np.isfinite(previous[column]):
            return "ESPERAR", 0, [
                "No hay suficientes datos para confirmar los cruces EMA."
            ]

    # Indicador 1: EMA 9 frente a EMA 21.
    ema_up = row["ema9"] > row["ema21"]
    ema_down = row["ema9"] < row["ema21"]

    if ema_up:
        score += 1
        reasons.append("EMA 9 sobre EMA 21: ALCISTA")
    elif ema_down:
        score -= 1
        reasons.append("EMA 9 debajo de EMA 21: BAJISTA")
    else:
        reasons.append("EMA 9 y EMA 21 neutrales")

    # Indicador 2: tendencia EMA 21 frente a EMA 50.
    trend_up = row["ema21"] > row["ema50"]
    trend_down = row["ema21"] < row["ema50"]

    if trend_up:
        score += 1
        reasons.append("EMA 21 sobre EMA 50: TENDENCIA ALCISTA")
    elif trend_down:
        score -= 1
        reasons.append("EMA 21 debajo de EMA 50: TENDENCIA BAJISTA")
    else:
        reasons.append("Tendencia EMA neutral")

    # Indicador 3: MACD.
    macd_up = row["macd"] > row["macd_signal"]
    macd_down = row["macd"] < row["macd_signal"]

    if macd_up:
        score += 1
        reasons.append("MACD alcista")
    elif macd_down:
        score -= 1
        reasons.append("MACD bajista")
    else:
        reasons.append("MACD neutral")

    # Indicador 4: momentum de 3 minutos.
    momentum = row["momentum3"]
    momentum_up = momentum > 0
    momentum_down = momentum < 0

    if momentum_up:
        score += 1
        reasons.append(f"Impulso reciente positivo: {momentum:+.4f}%")
    elif momentum_down:
        score -= 1
        reasons.append(f"Impulso reciente negativo: {momentum:+.4f}%")
    else:
        reasons.append("Impulso neutral")

    # RSI: filtro adicional para evitar entradas extremas.
    rsi = float(row["rsi"])
    reasons.append(f"RSI: {rsi:.1f}")

    rsi_up = 50 < rsi < 70
    rsi_down = 30 < rsi < 50

    if rsi >= 70:
        reasons.append("RSI alto: bloquea la señal UP")
    elif rsi <= 30:
        reasons.append("RSI bajo: bloquea la señal DOWN")
    elif rsi == 50:
        reasons.append("RSI neutral: no confirma la dirección")

    # Detectar cruces recientes de EMA.
    bearish_cross = (
        previous["ema9"] >= previous["ema21"]
        and row["ema9"] < row["ema21"]
    )

    bullish_cross = (
        previous["ema9"] <= previous["ema21"]
        and row["ema9"] > row["ema21"]
    )

    # Los 4 indicadores deben confirmar la misma dirección.
    confirm_up = (
        ema_up
        and trend_up
        and macd_up
        and momentum_up
    )

    confirm_down = (
        ema_down
        and trend_down
        and macd_down
        and momentum_down
    )

    # Señal UP: 4/4, RSI válido y sin cruce bajista.
    if confirm_up and rsi_up and not bearish_cross:
        signal = "COMPRAR UP"
        reasons.append(
            "CONFIRMACIÓN COMPLETA: 4/4 indicadores alcistas."
        )
        reasons.append("RSI confirmado y sin cruce contrario.")

    # Señal DOWN: 4/4, RSI válido y sin cruce alcista.
    elif confirm_down and rsi_down and not bullish_cross:
        signal = "COMPRAR DOWN"
        reasons.append(
            "CONFIRMACIÓN COMPLETA: 4/4 indicadores bajistas."
        )
        reasons.append("RSI confirmado y sin cruce contrario.")

    # Si falla cualquier condición, no dar señal de compra.
    else:
        signal = "ESPERAR"
        reasons.append(
            "No se cumplen todas las condiciones: se requiere "
            "confirmación 4/4, RSI válido y ausencia de cruce contrario."
        )

    # Avisos de cruces recientes.
    if bearish_cross:
        reasons.append("🔔 ALERTA: cruce reciente hacia bajada")

    if bullish_cross:
        reasons.append("🔔 ALERTA: cruce reciente hacia subida")

    return signal, score, reasons


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
        df = add_indicators(loader())
    except Exception as exc:
        st.error(f"Error cargando la gráfica: {exc}")
        st.info(
            "Comprueba tu conexión. Si es EUR/USD, el proveedor "
            "puede no estar respondiendo o haber cambiado su formato."
        )
        return

    seconds_left = None

    if market:
        seconds_left = max(
            0, int(market["_close_ts"] - time.time())
        )

        st.markdown("### Mercado de Kalshi")
        st.write("**Título:**", market.get("title", "No disponible"))
        st.write("**Ticker:**", market.get("ticker", "No disponible"))

        st.metric(
            "Cuenta regresiva",
            f"{seconds_left // 60:02d}:{seconds_left % 60:02d}",
        )

        probability = get_yes_probability(market)

        if np.isfinite(probability):
            st.metric("YES / UP implícito", f"{probability:.1f}%")
            st.metric("NO / DOWN implícito", f"{100-probability:.1f}%")
        else:
            st.info("No hay cotización disponible para calcular la probabilidad.")

        columns = st.columns(4)

        quote_fields = [
            ("YES bid", ("yes_bid_dollars", "yes_bid")),
            ("YES ask", ("yes_ask_dollars", "yes_ask")),
            ("NO bid", ("no_bid_dollars", "no_bid")),
            ("NO ask", ("no_ask_dollars", "no_ask")),
        ]

        for column, (label, fields) in zip(columns, quote_fields):
            value = price_cents(market, *fields)
            column.metric(
                label,
                f"{value:.1f}¢" if np.isfinite(value) else "N/D",
            )

        st.write("**Strike detectado:**", get_strike(market))
        st.caption(
            "Verifica las reglas del contrato: el strike mostrado "
            "no necesariamente describe toda la liquidación."
        )
    else:
        st.warning(
            f"No se encontró un contrato abierto para {series}. "
            "La gráfica de referencia puede seguir funcionando, "
            "pero no confirma un contrato activo."
        )

    signal, score, reasons = make_signal(df, seconds_left)

    st.markdown("### Señal del bot")

    if signal == "COMPRAR UP":
        st.success("🟢 " + signal)
    elif signal == "COMPRAR DOWN":
        st.error("🔴 " + signal)
    elif signal.startswith("POSIBLE"):
        st.warning("🟡 " + signal)
    else:
        st.info("⚪ " + signal)

    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)

    c1.metric("Precio de referencia", formatter(float(last["close"])))
    c2.metric("Puntuación técnica", f"{score:+d} / 4")
    c3.metric("RSI", f"{last['rsi']:.1f}")

    st.markdown("**Motivos de la señal**")
    for reason in reasons:
        st.write("• " + reason)

    # Aviso visible si cambia la señal entre actualizaciones.
    state_key = f"previous_signal_{series}"
    old_signal = st.session_state.get(state_key)

    if old_signal is not None and old_signal != signal:
        st.warning(f"🔔 CAMBIO DE SEÑAL: {old_signal} → {signal}")

    st.session_state[state_key] = signal

    # Gráfica de velas y medias móviles.
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
        go.Scatter(x=df["time"], y=df["ema9"], name="EMA 9")
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["ema21"], name="EMA 21")
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["ema50"], name="EMA 50")
    )

    fig.update_layout(
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(l=5, r=5, t=25, b=5),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "BTC usa Coinbase; EUR/USD usa Biquote. Estas referencias "
        "pueden diferir del índice de liquidación de Kalshi. "
        "La caché y el proveedor pueden retrasar los datos."
    )


# =====================================================
# PESTAÑAS PRINCIPALES
# =====================================================
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
    "Modo informativo: no ejecuta órdenes. Las reglas técnicas "
    "no han sido validadas mediante backtesting. Una señal DOWN "
    "no garantiza que el contrato vaya a ganar."
)
