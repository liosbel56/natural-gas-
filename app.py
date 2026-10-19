
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="pulse_refresh")
except ImportError:
    pass

# ==================================================
# CONFIGURACION
# ==================================================
st.set_page_config(
    page_title="PULSE AI 15M",
    page_icon="💜",
    layout="wide"
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "PulseAI/1.0"}

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#10081f,#21103d,#32175b);
    color: white;
}
h1,h2,h3,p,label,span {
    color: white;
}
div[data-testid="stMetric"] {
    background: #2a1747;
    border: 1px solid #7146a5;
    border-radius: 14px;
    padding: 12px;
}
.signal {
    padding: 20px 8px;
    margin: 10px 0;
    border-radius: 18px;
    text-align: center;
    font-size: 34px;
    font-weight: 900;
}
.up { background: #087f5b; color: white; }
.down { background: #b42342; color: white; }
.wait { background: #59427d; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("💜 PULSE AI")
st.caption("BTC/USD + EUR/USD | Análisis de 15 minutos")
st.warning(
    "Herramienta informativa. No ejecuta operaciones. "
    "Las señales pueden fallar y no garantizan beneficios."
)

# ==================================================
# CRONOMETRO INDEPENDIENTE
# ==================================================
now = datetime.now(timezone.utc)
seconds_elapsed = (now.minute % 15) * 60 + now.second
seconds_remaining = 900 - seconds_elapsed

st.markdown("## ⏱️ Ventana actual de 15 minutos")
tc1, tc2, tc3 = st.columns(3)

tc1.metric(
    "Tiempo restante",
    f"{seconds_remaining // 60:02d}:{seconds_remaining % 60:02d}"
)
tc2.metric("Hora UTC", now.strftime("%H:%M:%S"))
tc3.metric(
    "Próximo intervalo UTC",
    (now.replace(second=0, microsecond=0).minute // 15 * 15 + 15) % 60
    if now.minute // 15 < 3 else "00"
)

st.caption(
    "El cronómetro se basa en intervalos UTC de 15 minutos. "
    "El cierre real del contrato puede ser diferente."
)

# ==================================================
# FUENTE DE PRECIOS
# ==================================================
@st.cache_data(ttl=4)
def get_btc():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    r = requests.get(
        url,
        params={"granularity": 60},
        headers=HEADERS,
        timeout=12
    )
    r.raise_for_status()
    raw = r.json()

    if not raw:
        raise ValueError("Coinbase no devolvió velas de BTC.")

    df = pd.DataFrame(
        raw,
        columns=["time", "low", "high", "open", "close", "volume"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates("time")
    return df.tail(200).reset_index(drop=True)


@st.cache_data(ttl=5)
def get_eurusd():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"
    r = requests.get(
        url,
        params={"range": "1d", "interval": "1m"},
        headers=HEADERS,
        timeout=12
    )
    r.raise_for_status()
    payload = r.json()["chart"]

    if payload.get("error") or not payload.get("result"):
        raise ValueError("Yahoo Finance no devolvió datos de EUR/USD.")

    result = payload["result"][0]
    quotes = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(result["timestamp"], unit="s", utc=True),
        "open": quotes["open"],
        "high": quotes["high"],
        "low": quotes["low"],
        "close": quotes["close"]
    })
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.sort_values("time").tail(200).reset_index(drop=True)


# ==================================================
# CONTRATOS PUBLICOS KALSHI
# ==================================================
@st.cache_data(ttl=5)
def get_kalshi_markets(series):
    url = f"{KALSHI_API}/markets"
    r = requests.get(
        url,
        params={
            "series_ticker": series,
            "status": "open",
            "limit": 100
        },
        headers=HEADERS,
        timeout=12
    )
    r.raise_for_status()
    return r.json().get("markets", [])


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None


def find_current_market(series):
    markets = get_kalshi_markets(series)
    now = datetime.now(timezone.utc)
    candidates = []

    for market in markets:
        close_dt = parse_time(
            market.get("close_time")
            or market.get("expiration_time")
        )

        if close_dt and close_dt > now:
            market["_parsed_close"] = close_dt
            candidates.append(market)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["_parsed_close"])
    return candidates[0]


def extract_target(market):
    if not market:
        return None

    for key in (
        "floor_strike",
        "cap_strike",
        "strike",
        "custom_strike"
    ):
        value = market.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    text = " ".join(
        str(market.get(key, ""))
        for key in ("title", "subtitle", "yes_sub_title")
    )

    patterns = [
        r"\$\s*([\d,]+(?:\.\d+)?)",
        r"([\d,]+\.\d+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

    return None


def get_yes_price_cents(market):
    if not market:
        return None

    for key in ("yes_ask_dollars", "yes_bid_dollars"):
        value = market.get(key)
        if value is not None:
            try:
                return round(float(value) * 100, 1)
            except (ValueError, TypeError):
                pass

    for key in ("yes_ask", "yes_bid", "last_price"):
        value = market.get(key)
        if value is not None:
            try:
                return round(float(value), 1)
            except (ValueError, TypeError):
                pass

    return None


# ==================================================
# INDICADORES Y SEÑALES
# ==================================================
def analyze(df):
    df = df.copy()
    close = df["close"].astype(float)

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["bb_mid"] = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * deviation
    df["bb_lower"] = df["bb_mid"] - 2 * deviation

    if len(df) < 5:
        raise ValueError("No hay suficientes velas para analizar.")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    if last["ema9"] > last["ema21"]:
        score += 1
        reasons.append("EMA9 > EMA21")
    else:
        score -= 1
        reasons.append("EMA9 < EMA21")

    if last["ema21"] > last["ema50"]:
        score += 1
    elif last["ema21"] < last["ema50"]:
        score -= 1

    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD alcista")
    else:
        score -= 1
        reasons.append("MACD bajista")

    if last["rsi"] > 52:
        score += 1
    elif last["rsi"] < 48:
        score -= 1

    recent_change = (
        (close.iloc[-1] / close.iloc[-4] - 1) * 100
    )

    if recent_change > 0.015:
        score += 1
        reasons.append("Impulso reciente al alza")
    elif recent_change < -0.015:
        score -= 1
        reasons.append("Impulso reciente a la baja")

    extended_up = (
        last["rsi"] > 76
        or (
            pd.notna(last["bb_upper"])
            and close.iloc[-1] > last["bb_upper"]
        )
    )
    extended_down = (
        last["rsi"] < 24
        or (
            pd.notna(last["bb_lower"])
            and close.iloc[-1] < last["bb_lower"]
        )
    )

    reversal_down = (
        prev["macd"] > prev["macd_signal"]
        and last["macd"] < last["macd_signal"]
    ) or (
        prev["ema9"] > prev["ema21"]
        and last["ema9"] < last["ema21"]
    )

    reversal_up = (
        prev["macd"] < prev["macd_signal"]
        and last["macd"] > last["macd_signal"]
    ) or (
        prev["ema9"] < prev["ema21"]
        and last["ema9"] > last["ema21"]
    )

    if score >= 2 and not extended_up:
        signal, css = "BUY UP", "up"
    elif score <= -2 and not extended_down:
        signal, css = "BUY DOWN", "down"
    else:
        signal, css = "ESPERAR", "wait"

    if reversal_down:
        warning = "⚠️ Posible giro bajista"
    elif reversal_up:
        warning = "⚠️ Posible giro alcista"
    elif extended_up:
        warning = "⚠️ Movimiento alcista extendido: riesgo de entrar tarde"
    elif extended_down:
        warning = "⚠️ Movimiento bajista extendido: riesgo de rebote"
    else:
        warning = "Sin giro confirmado"

    # Extrapolación sencilla, no un modelo entrenado.
    recent = close.tail(15).to_numpy()
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    projected = float(close.iloc[-1] + slope * 15)

    return df, {
        "price": float(close.iloc[-1]),
        "projected": projected,
        "score": score,
        "signal": signal,
        "css": css,
        "rsi": float(last["rsi"]),
        "macd": float(last["macd"]),
        "ema9": float(last["ema9"]),
        "ema21": float(last["ema21"]),
        "ema50": float(last["ema50"]),
        "change": float(recent_change),
        "warning": warning,
        "reasons": reasons
    }


# ==================================================
# INTERFAZ DE CADA ACTIVO
# ==================================================
def show_asset(title, df, market, is_btc):
    df, result = analyze(df)

    st.markdown(f"## {title}")
    st.markdown(
        f'<div class="signal {result["css"]}">'
        f'{result["signal"]}</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio actual", f'{result["price"]:,.5f}')
    c2.metric("Estimación +15 min", f'{result["projected"]:,.5f}')
    c3.metric("RSI", f'{result["rsi"]:.1f}')

    st.write(f'**EMA 9:** {result["ema9"]:.5f}')
    st.write(f'**EMA 21:** {result["ema21"]:.5f}')
    st.write(f'**EMA 50:** {result["ema50"]:.5f}')
    st.write(f'**MACD:** {result["macd"]:.6f}')
    st.write(f'**Cambio reciente:** {result["change"]:.4f}%')

    if market:
        st.write(
            f'**Contrato:** {market.get("title", "Sin título")}'
        )
        st.code(market.get("ticker", "Ticker no disponible"))

        target = extract_target(market)

        if target is not None:
            st.write(f"**Objetivo detectado:** {target:,.5f}")

            if is_btc:
                if result["projected"] > target:
                    st.info("Estimación por encima del objetivo.")
                elif result["projected"] < target:
                    st.info("Estimación por debajo del objetivo.")
                else:
                    st.info("Estimación cerca del objetivo.")
        else:
            st.warning(
                "No se pudo verificar el objetivo exacto del contrato."
            )

        cents = get_yes_price_cents(market)
        if cents is not None:
            st.metric("Precio YES", f"{cents:.1f}¢")

        close_dt = market.get("_parsed_close")
        if close_dt:
            seconds = max(
                0,
                int((close_dt - datetime.now(timezone.utc)).total_seconds())
            )
            st.metric(
                "Cierre del contrato",
                f"{seconds // 60:02d}:{seconds % 60:02d}"
            )
    else:
        st.warning(
            "No se encontró contrato abierto. "
            "El análisis de precio continúa disponible."
        )

    st.info(result["warning"])
    st.caption("Indicadores: " + " · ".join(result["reasons"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["close"],
        mode="lines", name="Precio"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema9"],
        mode="lines", name="EMA 9"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema21"],
        mode="lines", name="EMA 21"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1c1033",
        plot_bgcolor="#1c1033",
        height=330,
        margin=dict(l=5, r=5, t=20, b=5),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================================================
# EJECUCION
# ==================================================
left, right = st.columns(2)

with left:
    try:
        btc_df = get_btc()
        btc_market = find_current_market("KXBTC15M")
        show_asset("₿ BITCOIN / USD", btc_df, btc_market, True)
    except Exception as exc:
        st.error(f"Error en BTC/USD: {exc}")

with right:
    try:
        eur_df = get_eurusd()
        eur_market = find_current_market("KXEURUSD15M")
        show_asset("💱 EUR/USD", eur_df, eur_market, False)
    except Exception as exc:
        st.error(f"Error en EUR/USD: {exc}")

st.divider()
st.caption(
    "Actualización automática aproximada cada 5 segundos. "
    "Las cotizaciones externas pueden diferir de las usadas para liquidar "
    "Kalshi. Las señales no son asesoramiento financiero."
)
