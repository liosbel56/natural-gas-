
import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="refresh")
except ImportError:
    pass

# =========================
# CONFIGURACION
# =========================
st.set_page_config(
    page_title="Pulse AI | 15M",
    page_icon="💜",
    layout="wide"
)

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "PulseAI/1.0"}

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#160d2d,#251047,#100b20);
    color: white;
}
h1,h2,h3,p,label,span {
    color: white;
}
div[data-testid="stMetric"] {
    background: #2b1850;
    border: 1px solid #6941a5;
    padding: 14px;
    border-radius: 14px;
}
.signal {
    padding: 22px 10px;
    border-radius: 18px;
    text-align: center;
    font-size: 34px;
    font-weight: 900;
    margin: 10px 0;
}
.up { background: #087f5b; color: white; }
.down { background: #b42342; color: white; }
.wait { background: #59427d; color: white; }
.small { color: #d7c6f5; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.title("💜 PULSE AI")
st.caption("BTC + EUR/USD | Analizador de ventanas de 15 minutos")
st.warning(
    "Solo señales informativas. No ejecuta operaciones. "
    "Las predicciones pueden fallar y no garantizan ganancias."
)

# =========================
# DATOS DE PRECIOS
# =========================
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
    data = r.json()

    df = pd.DataFrame(
        data,
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
    result = r.json()["chart"]["result"][0]

    quotes = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "time": pd.to_datetime(result["timestamp"], unit="s", utc=True),
        "open": quotes["open"],
        "high": quotes["high"],
        "low": quotes["low"],
        "close": quotes["close"],
        "volume": quotes.get("volume")
    })
    df = df.dropna(subset=["close"]).sort_values("time")
    return df.tail(200).reset_index(drop=True)


# =========================
# MERCADOS PUBLICOS KALSHI
# =========================
@st.cache_data(ttl=5)
def get_kalshi_market(series):
    url = f"{KALSHI}/markets"
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
    markets = r.json().get("markets", [])

    now = datetime.now(timezone.utc)
    valid = []

    for m in markets:
        close_str = m.get("close_time") or m.get("expiration_time")
        if not close_str:
            continue
        try:
            close_dt = datetime.fromisoformat(
                close_str.replace("Z", "+00:00")
            )
            if close_dt > now:
                m["_close_dt"] = close_dt
                valid.append(m)
        except (ValueError, TypeError):
            continue

    if not valid:
        return None

    # El contrato abierto que cierra primero.
    valid.sort(key=lambda m: m["_close_dt"])
    return valid[0]


def get_target(market):
    if not market:
        return None

    for key in (
        "floor_strike", "cap_strike", "strike",
        "custom_strike"
    ):
        value = market.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    text = " ".join(str(market.get(k, "")) for k in (
        "title", "subtitle", "yes_sub_title"
    ))

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


def market_price_cents(market, side="yes"):
    if not market:
        return None

    keys = (
        [f"{side}_ask_dollars", f"{side}_bid_dollars"]
        if side == "yes"
        else ["no_ask_dollars", "no_bid_dollars"]
    )

    for key in keys:
        value = market.get(key)
        if value is not None:
            try:
                return round(float(value) * 100, 1)
            except (TypeError, ValueError):
                pass

    # Campos antiguos de Kalshi están en centavos.
    for key in (
        f"{side}_ask", f"{side}_bid",
        "last_price"
    ):
        value = market.get(key)
        if value is not None:
            try:
                return round(float(value), 1)
            except (TypeError, ValueError):
                pass

    return None


# =========================
# INDICADORES TECNICOS
# =========================
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
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    previous = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous).abs(),
        (df["low"] - previous).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    # Estimación sencilla: extrapola el cambio medio
    # de los últimos minutos, no una predicción segura.
    recent = close.tail(15).to_numpy()
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0] if len(recent) >= 5 else 0
    projected = float(close.iloc[-1] + slope * 15)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    if last["ema9"] > last["ema21"]:
        score += 1
        reasons.append("EMA9 sobre EMA21")
    else:
        score -= 1
        reasons.append("EMA9 bajo EMA21")

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

    # Movimiento de precio reciente: ayuda a detectar
    # el impulso temprano sin esperar una gran vela.
    change = (
        (close.iloc[-1] / close.iloc[-4]) - 1
    ) * 100 if len(close) >= 4 else 0

    if change > 0.015:
        score += 1
        reasons.append("Impulso reciente positivo")
    elif change < -0.015:
        score -= 1
        reasons.append("Impulso reciente negativo")

    # Evita comprar una subida ya muy extendida.
    extended_up = (
        last["rsi"] > 76 or
        (pd.notna(last["bb_upper"]) and close.iloc[-1] > last["bb_upper"])
    )
    extended_down = (
        last["rsi"] < 24 or
        (pd.notna(last["bb_lower"]) and close.iloc[-1] < last["bb_lower"])
    )

    # Alerta de giro: pérdida de momentum reciente.
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

    if score >= 3 and not extended_up:
        signal = "BUY UP"
        css = "up"
    elif score <= -3 and not extended_down:
        signal = "BUY DOWN"
        css = "down"
    else:
        signal = "ESPERAR"
        css = "wait"

    if reversal_down:
        warning = "⚠️ Posible giro bajista: vigilar el impulso."
    elif reversal_up:
        warning = "⚠️ Posible giro alcista: vigilar el impulso."
    elif extended_up:
        warning = "⚠️ Precio extendido al alza: riesgo de entrar tarde."
    elif extended_down:
        warning = "⚠️ Precio extendido a la baja: riesgo de rebote."
    else:
        warning = "Sin giro confirmado por los indicadores."

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
        "change": float(change),
        "warning": warning,
        "reasons": reasons
    }


# =========================
# PANEL DE CADA MERCADO
# =========================
def show_asset(title, df, market, is_btc):
    df, a = analyze(df)

    st.markdown(f"## {title}")
    st.markdown(
        f'<div class="signal {a["css"]}">{a["signal"]}</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio actual", f'{a["price"]:,.5f}')
    c2.metric("Estimación a 15 min", f'{a["projected"]:,.5f}')
    c3.metric("RSI", f'{a["rsi"]:.1f}')

    st.write(f'**EMA 9:** {a["ema9"]:.5f}')
    st.write(f'**EMA 21:** {a["ema21"]:.5f}')
    st.write(f'**EMA 50:** {a["ema50"]:.5f}')
    st.write(f'**MACD:** {a["macd"]:.6f}')
    st.write(f'**Cambio reciente:** {a["change"]:.4f}%')

    if market:
        st.write(f'**Contrato Kalshi:** {market.get("title", market.get("ticker", "Sin título"))}')
        st.write(f'**Ticker:** `{market.get("ticker", "No disponible")}`')

        target = get_target(market)
        if target is not None:
            st.write(f'**Objetivo detectado:** {target:,.5f}')
            if is_btc:
                diff = a["projected"] - target
                if diff > 0:
                    st.info("Estimación por encima del objetivo.")
                elif diff < 0:
                    st.info("Estimación por debajo del objetivo.")
                else:
                    st.info("Estimación cerca del objetivo.")
        else:
            st.warning(
                "No pude verificar el objetivo exacto en los campos "
                "del contrato. No se inventará un objetivo."
            )

        price = market_price_cents(market, "yes")
        if price is not None:
            st.metric("Precio YES disponible", f"{price:.1f}¢")
        else:
            st.caption("Precio YES no disponible en esta respuesta.")

        close_dt = market.get("_close_dt")
        if close_dt:
            seconds = max(0, int(
                (close_dt - datetime.now(timezone.utc)).total_seconds()
            ))
            st.metric(
                "Tiempo restante",
                f"{seconds // 60:02d}:{seconds % 60:02d}"
            )
    else:
        st.warning(
            "No se encontró un contrato abierto para esta serie. "
            "Los indicadores de precio siguen disponibles."
        )

    st.info(a["warning"])
    st.caption("Confirmaciones: " + " · ".join(a["reasons"]))

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
        height=350,
        margin=dict(l=5, r=5, t=20, b=5),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# EJECUCION
# =========================
left, right = st.columns(2)

with left:
    try:
        btc_df = get_btc()
        btc_market = get_kalshi_market("KXBTC15M")
        show_asset("₿ BITCOIN · 15 MIN", btc_df, btc_market, True)
    except Exception as e:
        st.error(f"Error en BTC: {e}")

with right:
    try:
        eur_df = get_eurusd()
        eur_market = get_kalshi_market("KXEURUSD15M")
        show_asset("💱 EUR/USD · 15 MIN", eur_df, eur_market, False)
    except Exception as e:
        st.error(f"Error en EUR/USD: {e}")

st.divider()
st.caption(
    "Actualización automática cada 5 segundos cuando está disponible. "
    "Las señales son heurísticas y no garantizan resultados. "
    "Verifica siempre el contrato y su regla de liquidación en Kalshi."
)
