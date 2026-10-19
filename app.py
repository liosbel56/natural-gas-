
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="Trading Signals Bot",
    page_icon="📈",
    layout="wide"
)

BIQUOTE = "https://biquote.io/api"
COINBASE = "https://api.exchange.coinbase.com"

# ===== ESTILO MORADO =====
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #10002b, #240046, #16002e);
    color: #f5eaff;
}
header[data-testid="stHeader"] {
    background: rgba(16, 0, 43, 0.9);
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}
div[data-testid="stMetric"],
div[data-testid="stSelectbox"] {
    background: rgba(62, 15, 112, 0.55);
    border: 1px solid #7b2cff;
    border-radius: 14px;
    padding: 12px;
}
h1, h2, h3, p, label {
    color: #f5eaff !important;
}
div[data-testid="stAlert"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 TRADING SIGNALS BOT")
st.caption("BTC/USD + EUR/USD | Análisis técnico de 15 minutos")


def get_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        timeout=20,
        headers={"User-Agent": "TradingSignalsBot/1.0"}
    )
    response.raise_for_status()
    return response.json()


def get_eurusd_candles(interval):
    data = get_json(
        f"{BIQUOTE}/EURUSD/ohlc",
        {"interval": interval, "limit": 200}
    )

    bars = data.get("bars") if isinstance(data, dict) else None
    if not bars:
        raise ValueError("BiQuote no devolvió velas EUR/USD.")

    df = pd.DataFrame(bars)

    if "openTime" not in df.columns:
        raise ValueError("Falta openTime en EUR/USD.")

    df["openTime"] = pd.to_datetime(
        df["openTime"], utc=True
    )
    df = df.set_index("openTime").sort_index()

    if "isOpen" in df.columns:
        df = df[~df["isOpen"].fillna(False)]

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(
        subset=["open", "high", "low", "close"]
    )


def get_btc_candles(granularity):
    now = int(datetime.now(timezone.utc).timestamp())
    end_ts = (now // granularity) * granularity
    start_ts = end_ts - granularity * 200

    def iso(ts):
        return datetime.fromtimestamp(
            ts, timezone.utc
        ).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Coinbase limita el número de velas por petición.
    # Descargamos en bloques de 250 o menos.
    chunks = []
    chunk_size = granularity * 250
    cursor = start_ts

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_size, end_ts)

        data = get_json(
            f"{COINBASE}/products/BTC-USD/candles",
            {
                "granularity": granularity,
                "start": iso(cursor),
                "end": iso(chunk_end)
            }
        )

        if isinstance(data, list) and data:
            chunks.extend(data)

        cursor = chunk_end

    if not chunks:
        raise ValueError("Coinbase no devolvió velas BTC/USD.")

    df = pd.DataFrame(
        chunks,
        columns=[
            "time", "low", "high", "open", "close", "volume"
        ]
    )

    df = df.drop_duplicates(subset=["time"])
    df["openTime"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )
    df = df.set_index("openTime").sort_index()

    # Solo velas cerradas
    df = df[df.index.astype("int64") // 10**9 < end_ts]

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(
        subset=["open", "high", "low", "close"]
    )


def get_price(market):
    if market == "BTC/USD":
        data = get_json(
            f"{COINBASE}/products/BTC-USD/ticker"
        )
        return float(data["price"]), False

    data = get_json(f"{BIQUOTE}/EURUSD")
    price = data.get("mid")

    if price is None:
        raise ValueError("No hay precio EUR/USD disponible.")

    stale = (
        data.get("stale") is True
        or data.get("marketState") == "closed"
    )
    return float(price), stale


def add_indicators(df):
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(
        span=9, adjust=False
    ).mean()
    df["ema21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = 100 - 100 / (1 + rs)
    df.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi"] = 50

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )
    df["macd_hist"] = (
        macd - macd.ewm(span=9, adjust=False).mean()
    )

    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr = tr.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    plus_di = 100 * plus_dm.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean() / atr.replace(0, float("nan"))

    minus_di = 100 * minus_dm.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean() / atr.replace(0, float("nan"))

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, float("nan"))

    df["adx"] = dx.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    return df


def score_latest(df):
    row = df.iloc[-1]
    score = 0

    if row["ema9"] > row["ema21"]:
        score += 1
    elif row["ema9"] < row["ema21"]:
        score -= 1

    if pd.notna(row["rsi"]):
        if row["rsi"] >= 55:
            score += 1
        elif row["rsi"] <= 45:
            score -= 1

    if row["macd_hist"] > 0:
        score += 1
    elif row["macd_hist"] < 0:
        score -= 1

    if pd.notna(row["bb_mid"]):
        if row["close"] > row["bb_mid"]:
            score += 1
        elif row["close"] < row["bb_mid"]:
            score -= 1

    if pd.notna(row["adx"]) and row["adx"] >= 18:
        if row["plus_di"] > row["minus_di"]:
            score += 1
        elif row["plus_di"] < row["minus_di"]:
            score -= 1

    return score


def show_signal(signal):
    if signal == "BUY UP":
        color = "#00ff9d"
        background = "#062c27"
        icon = "⬆️"
        description = "POSIBLE MOVIMIENTO ALCISTA"
    elif signal == "BUY DOWN":
        color = "#ff4268"
        background = "#3b1028"
        icon = "⬇️"
        description = "POSIBLE MOVIMIENTO BAJISTA"
    else:
        color = "#ffc857"
        background = "#382508"
        icon = "⏳"
        description = "ESPERANDO CONFIRMACIÓN"

    st.markdown(
        f"""
        <div style="
            background:{background};
            border:3px solid {color};
            box-shadow:0 0 18px {color}55;
            border-radius:20px;
            padding:25px 8px;
            text-align:center;
            margin:12px 0 20px 0;
        ">
          <div style="
              color:{color};
              font-size:clamp(38px, 9vw, 62px);
              font-weight:900;
              line-height:1.2;
          ">{icon} {signal}</div>
          <div style="
              color:white;
              font-size:16px;
              font-weight:700;
              margin-top:12px;
          ">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_bot(market):
    st.subheader(f"Mercado: {market}")

    try:
        if market == "BTC/USD":
            df1 = get_btc_candles(60)
            df15 = get_btc_candles(900)
        else:
            df1 = get_eurusd_candles("1m")
            df15 = get_eurusd_candles("15m")

        price, stale = get_price(market)

        if len(df1) < 60 or len(df15) < 60:
            st.warning(
                "No hay suficientes velas cerradas para calcular "
                "los indicadores. Intenta más tarde."
            )
            return

        df1 = add_indicators(df1)
        df15 = add_indicators(df15)

        s1 = score_latest(df1)
        s15 = score_latest(df15)

        if stale:
            signal = "ESPERAR"
            explanation = "El precio de EUR/USD puede estar cerrado o desactualizado."
        elif s15 >= 3 and s1 >= 1:
            signal = "BUY UP"
            explanation = "Los indicadores favorecen una posible subida; no es garantía."
        elif s15 <= -3 and s1 <= -1:
            signal = "BUY DOWN"
            explanation = "Los indicadores favorecen una posible bajada; no es garantía."
        else:
            signal = "ESPERAR"
            explanation = "Los indicadores no coinciden lo suficiente."

        a, b = st.columns(2)
        a.metric(
            "PRECIO ACTUAL",
            f"${price:,.2f}" if market == "BTC/USD"
            else f"{price:.5f}"
        )
        b.metric("HORIZONTE", "15 minutos")

        show_signal(signal)
        st.write(explanation)

        a, b = st.columns(2)
        a.metric("Puntuación 1 minuto", f"{s1} / 5")
        b.metric("Puntuación 15 minutos", f"{s15} / 5")

        row = df15.iloc[-1]
        st.subheader("Indicadores de 15 minutos")

        a, b, c, d = st.columns(4)
        a.metric(
            "RSI (14)",
            f"{row['rsi']:.1f}" if pd.notna(row["rsi"]) else "N/D"
        )
        b.metric("MACD", f"{row['macd_hist']:.6f}")
        c.metric(
            "ADX (14)",
            f"{row['adx']:.1f}" if pd.notna(row["adx"]) else "N/D"
        )
        d.metric(
            "Bollinger",
            "Sobre media" if row["close"] > row["bb_mid"]
            else "Bajo media"
        )

        st.subheader(f"Gráfica de velas de 15 minutos — {market}")
        chart = df15.tail(80)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart.index,
            open=chart["open"],
            high=chart["high"],
            low=chart["low"],
            close=chart["close"],
            name=market
        ))
        fig.add_trace(go.Scatter(
            x=chart.index, y=chart["ema9"],
            name="EMA 9", line=dict(color="#00cfff", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=chart.index, y=chart["ema21"],
            name="EMA 21", line=dict(color="#ffb703", width=2)
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#190035",
            plot_bgcolor="#190035",
            font=dict(color="#f5eaff"),
            xaxis_rangeslider_visible=False,
            height=430,
            margin=dict(l=5, r=5, t=25, b=5)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🤖 Análisis del bot")
        st.info(explanation)

        st.markdown("#### Resumen de indicadores")
        checks = [
            ("EMA 9 frente a EMA 21", row["ema9"] > row["ema21"]),
            ("MACD positivo", row["macd_hist"] > 0),
            ("Precio sobre media Bollinger", row["close"] > row["bb_mid"]),
            ("ADX indica tendencia fuerte", row["adx"] >= 18 if pd.notna(row["adx"]) else False)
        ]

        for label, passed in checks:
            st.write(f"{'✅' if passed else '▫️'} {label}")

        st.caption(
            "Las señales son estimaciones técnicas. Los precios pueden cambiar "
            "y no se garantizan ganancias."
        )

    except Exception as e:
        st.error("No se pudieron cargar los datos del mercado.")
        st.caption(f"Detalle técnico: {e}")


market = st.selectbox(
    "ELIGE EL MERCADO",
    ["BTC/USD", "EUR/USD"]
)


@st.fragment(run_every="15s")
def live_panel(selected_market):
    render_bot(selected_market)


live_panel(market)

st.divider()
st.caption(
    "Bot experimental. No ejecuta operaciones automáticamente "
    "ni garantiza resultados."
)
