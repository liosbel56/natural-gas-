
import time
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Market Signals | BTC & EUR/USD",
    page_icon="💜",
    layout="wide"
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_API = "https://api.exchange.coinbase.com"
YAHOO_API = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"

# Actualizar la pantalla cada segundo para mover el cronómetro.
st_autorefresh(interval=1000, key="market_refresh")

# =========================================================
# ESTILO MORADO
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10051f, #241044, #120724);
    color: #f7f0ff;
}
[data-testid="stSidebar"] {
    background: #190b2e;
}
h1, h2, h3, p, label {
    color: #f7f0ff !important;
}
div[data-testid="stMetric"] {
    background: rgba(119, 55, 190, 0.18);
    border: 1px solid #7543a8;
    border-radius: 12px;
    padding: 12px;
}
.stButton > button {
    background: #7c3aed;
    color: white;
    border: 0;
    border-radius: 10px;
}
.signal-card, .timer-card {
    padding: 18px;
    border: 1px solid #8052b8;
    border-radius: 14px;
    background: rgba(92, 42, 143, 0.25);
    margin-bottom: 12px;
}
.timer-number {
    font-size: 42px;
    font-weight: 800;
    color: #d8b4fe;
    text-align: center;
}
.small-note {
    color: #d1b9ee;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CRONÓMETRO GENERAL DE 15 MINUTOS
# =========================================================
def mostrar_cronometro():
    ahora = datetime.now(timezone.utc)
    bloque_minuto = (ahora.minute // 15) * 15

    inicio = ahora.replace(
        minute=bloque_minuto,
        second=0,
        microsecond=0
    )
    final = inicio + timedelta(minutes=15)

    restantes = max(0, int((final - ahora).total_seconds()))
    mm, ss = divmod(restantes, 60)

    st.markdown(
        f"""
        <div class="timer-card">
          <p style="text-align:center;">⏱️ CRONÓMETRO · BLOQUE DE 15 MINUTOS</p>
          <div class="timer-number">{mm:02d}:{ss:02d}</div>
          <p style="text-align:center;">
            Tiempo restante del bloque actual (UTC)
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(
        f"Bloque actual: {inicio.strftime('%H:%M')}–"
        f"{final.strftime('%H:%M')} UTC. "
        "Comprueba la hora de cierre del contrato en Kalshi."
    )


# =========================================================
# DATOS BTC/USD
# =========================================================
@st.cache_data(ttl=15, show_spinner=False)
def get_btc_data():
    response = requests.get(
        f"{COINBASE_API}/products/BTC-USD/candles",
        params={"granularity": 60},
        timeout=12,
        headers={"User-Agent": "MarketSignals/1.0"}
    )
    response.raise_for_status()
    raw = response.json()

    if not raw:
        raise ValueError("Coinbase no devolvió velas de BTC.")

    df = pd.DataFrame(
        raw,
        columns=["time", "low", "high", "open", "close", "volume"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    for col in ["low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .dropna()
        .reset_index(drop=True)
    )

    if len(df) < 55:
        raise ValueError("No hay suficientes velas de BTC.")

    return df


# =========================================================
# DATOS EUR/USD
# =========================================================
@st.cache_data(ttl=30, show_spinner=False)
def get_eurusd_data():
    now = int(time.time())
    start = now - 5 * 24 * 60 * 60

    response = requests.get(
        YAHOO_API,
        params={
            "period1": start,
            "period2": now,
            "interval": "1m"
        },
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    payload = response.json()

    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError("No se recibieron datos de EUR/USD.")

    item = result[0]
    timestamps = item.get("timestamp", [])
    quotes = item.get("indicators", {}).get("quote", [{}])[0]

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quotes.get("open", []),
        "high": quotes.get("high", []),
        "low": quotes.get("low", []),
        "close": quotes.get("close", []),
        "volume": quotes.get("volume", [0] * len(timestamps))
    })

    df = (
        df.dropna(subset=["open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if len(df) < 55:
        raise ValueError("No hay suficientes velas de EUR/USD.")

    return df


# =========================================================
# INDICADORES TÉCNICOS
# =========================================================
def add_indicators(df):
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

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

    previous_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    return df


# =========================================================
# MOTOR DE SEÑALES
# Cada mercado se analiza por separado
# =========================================================
def analyze_market(df):
    df = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    up_reasons = []
    down_reasons = []

    if last["ema9"] > last["ema21"]:
        score += 1
        up_reasons.append("EMA 9 sobre EMA 21")
    else:
        score -= 1
        down_reasons.append("EMA 9 bajo EMA 21")

    if last["close"] > last["ema50"]:
        score += 1
        up_reasons.append("Precio sobre EMA 50")
    else:
        score -= 1
        down_reasons.append("Precio bajo EMA 50")

    if last["macd"] > last["macd_signal"]:
        score += 1
        up_reasons.append("MACD alcista")
    else:
        score -= 1
        down_reasons.append("MACD bajista")

    rsi = last["rsi"]
    if pd.notna(rsi):
        if 50 < rsi < 70:
            score += 1
            up_reasons.append(f"RSI favorable: {rsi:.1f}")
        elif 30 < rsi < 50:
            score -= 1
            down_reasons.append(f"RSI débil: {rsi:.1f}")

    change_5 = (
        last["close"] / df.iloc[-6]["close"] - 1
    ) * 100

    if change_5 > 0.08:
        score += 1
        up_reasons.append("Impulso positivo reciente")
    elif change_5 < -0.08:
        score -= 1
        down_reasons.append("Impulso negativo reciente")

    cross_up = (
        prev["ema9"] <= prev["ema21"]
        and last["ema9"] > last["ema21"]
    )
    cross_down = (
        prev["ema9"] >= prev["ema21"]
        and last["ema9"] < last["ema21"]
    )

    if score >= 4:
        signal = "POSIBLE SUBIDA"
        direction = "UP"
    elif score <= -4:
        signal = "POSIBLE BAJADA"
        direction = "DOWN"
    elif score >= 2:
        signal = "TENDENCIA ALCISTA DÉBIL"
        direction = "UP"
    elif score <= -2:
        signal = "TENDENCIA BAJISTA DÉBIL"
        direction = "DOWN"
    else:
        signal = "SIN CONFIRMACIÓN"
        direction = "WAIT"

    if cross_up:
        reversal = "Posible giro alcista: EMA 9 cruzó sobre EMA 21."
    elif cross_down:
        reversal = "Posible giro bajista: EMA 9 cruzó bajo EMA 21."
    elif change_5 > 0.12 and score < 0:
        reversal = "Rebote reciente con señales mixtas."
    elif change_5 < -0.12 and score > 0:
        reversal = "Caída reciente con señales mixtas."
    else:
        reversal = "Sin cruce reciente confirmado."

    return {
        "df": df,
        "price": float(last["close"]),
        "rsi": float(rsi) if pd.notna(rsi) else None,
        "score": score,
        "signal": signal,
        "direction": direction,
        "change_5": float(change_5),
        "reversal": reversal,
        "up_reasons": up_reasons,
        "down_reasons": down_reasons,
        "candle_time": last["time"]
    }


# =========================================================
# KALSHI: MERCADOS ABIERTOS
# =========================================================
@st.cache_data(ttl=20, show_spinner=False)
def get_kalshi_markets(series_ticker):
    response = requests.get(
        f"{KALSHI_API}/markets",
        params={
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 20
        },
        timeout=12,
        headers={"User-Agent": "MarketSignals/1.0"}
    )
    response.raise_for_status()
    return response.json().get("markets", [])


def show_kalshi(series_ticker):
    try:
        markets = get_kalshi_markets(series_ticker)
        if not markets:
            st.info(
                f"No se encontraron contratos abiertos para {series_ticker}. "
                "Comprueba el identificador de serie en Kalshi."
            )
            return

        rows = []
        for market in markets:
            rows.append({
                "Contrato": market.get("title", "—"),
                "Ticker": market.get("ticker", "—"),
                "YES bid": market.get("yes_bid_dollars", "—"),
                "YES ask": market.get("yes_ask_dollars", "—"),
                "Último precio": market.get("last_price_dollars", "—"),
                "Cierre": market.get("close_time", "—")
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )
    except Exception as exc:
        st.warning(f"No se pudieron cargar los contratos de Kalshi: {exc}")


# =========================================================
# GRÁFICA
# =========================================================
def show_chart(df, title):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=title
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ema9"],
        name="EMA 9",
        line=dict(color="#e879f9", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ema21"],
        name="EMA 21",
        line=dict(color="#a78bfa", width=1.5)
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,7,36,0.8)",
        height=460,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PRESENTACIÓN DEL MERCADO
# =========================================================
def show_market(data_function, name, kalshi_series, is_btc=False):
    try:
        raw = data_function()
        result = analyze_market(raw)
        df = result["df"]

        stale_seconds = (
            pd.Timestamp.now(tz="UTC") - result["candle_time"]
        ).total_seconds()

        c1, c2, c3, c4 = st.columns(4)

        if is_btc:
            c1.metric("BTC/USD", f"${result['price']:,.2f}")
        else:
            c1.metric("EUR/USD", f"{result['price']:.5f}")

        c2.metric(
            "RSI (14)",
            f"{result['rsi']:.1f}" if result["rsi"] is not None else "—"
        )
        c3.metric("Puntuación", f"{result['score']}/5")
        c4.metric("Cambio aprox. 5 min", f"{result['change_5']:+.3f}%")

        st.markdown(
            f'<div class="signal-card"><h3>{result["signal"]}</h3>'
            f'<p>{result["reversal"]}</p></div>',
            unsafe_allow_html=True
        )

        if stale_seconds > 180:
            st.warning("Los datos parecen antiguos. Verifica la fuente.")

        left, right = st.columns(2)

        with left:
            st.markdown("**Factores alcistas**")
            for reason in result["up_reasons"]:
                st.write(f"• {reason}")
            if not result["up_reasons"]:
                st.write("Sin factores alcistas confirmados.")

        with right:
            st.markdown("**Factores bajistas**")
            for reason in result["down_reasons"]:
                st.write(f"• {reason}")
            if not result["down_reasons"]:
                st.write("Sin factores bajistas confirmados.")

        show_chart(df.tail(180), f"{name} · Velas de 1 minuto")

        st.subheader(f"Contratos de {name} en Kalshi")
        show_kalshi(kalshi_series)

    except Exception as exc:
        st.error(f"No se pudieron cargar los datos de {name}: {exc}")


# =========================================================
# MENÚ PRINCIPAL
# =========================================================
st.title("💜 Market Signals")
st.caption("Análisis técnico experimental · No garantiza resultados")

with st.sidebar:
    st.title("💜 Menú")
    page = st.radio(
        "Selecciona el mercado",
        ["BTC/USD", "EUR/USD"]
    )

    st.divider()
    if st.button("Actualizar datos ahora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Cronómetro visible en las dos secciones
mostrar_cronometro()

if page == "BTC/USD":
    st.header("₿ Bitcoin / USD")
    show_market(
        get_btc_data,
        "BTC/USD",
        "KXBTC15M",
        is_btc=True
    )

elif page == "EUR/USD":
    st.header("€ Euro / USD")
    show_market(
        get_eurusd_data,
        "EUR/USD",
        "KXEURUSD15M",
        is_btc=False
    )

st.divider()
st.caption(
    "El cronómetro usa bloques de 15 minutos UTC; no verifica el cierre "
    "real de cada contrato. Confirma siempre el ticker, el precio y las "
    "reglas del mercado en Kalshi. Las señales son experimentales y no "
    "son una garantía de rentabilidad."
)
