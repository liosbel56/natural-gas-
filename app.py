
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================
# EUR/USD 15-MINUTE ANALYZER — SIGNALS ONLY
# Live source: BiQuote public API
# No automatic trading. No guaranteed prediction accuracy.
# =========================================================

st.set_page_config(
    page_title="EUR/USD AI Signal Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_URL = "https://biquote.io/api"
TIMEOUT = 10

# Refresh the dashboard every 5 seconds while it is open.
st_autorefresh(interval=5000, key="live_refresh")


# -------------------- STYLE --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(150deg, #100b24 0%, #17102f 55%, #0b1020 100%);
    color: #f5f3ff;
}
.block-container {
    max-width: 1250px;
    padding-top: 1.3rem;
}
h1, h2, h3, p, label {
    color: #f5f3ff !important;
}
.eyebrow {
    color: #b9a5ff;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 800;
}
.hero {
    background: linear-gradient(135deg, #28144b, #17102d);
    border: 1px solid #5a3b92;
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 15px;
}
.timer {
    color: #ff414f;
    font-size: clamp(52px, 10vw, 88px);
    font-weight: 900;
    line-height: 1.05;
    text-shadow: 0 0 24px #ff243a44;
    font-variant-numeric: tabular-nums;
}
.price {
    color: #ffffff;
    font-size: clamp(34px, 6vw, 52px);
    font-weight: 850;
}
.muted {
    color: #b6b0ce;
    font-size: 13px;
}
.metric {
    background: #21183b;
    border: 1px solid #443163;
    border-radius: 16px;
    padding: 18px;
    min-height: 105px;
}
.signal-up {
    background: #103d32;
    border: 1px solid #26d99a;
    border-radius: 18px;
    padding: 20px;
}
.signal-down {
    background: #451c30;
    border: 1px solid #ff526b;
    border-radius: 18px;
    padding: 20px;
}
.signal-wait {
    background: #302747;
    border: 1px solid #8c78b7;
    border-radius: 18px;
    padding: 20px;
}
.signal-title {
    font-size: 30px;
    font-weight: 900;
}
div[data-testid="stMetric"] {
    background: #21183b;
    border: 1px solid #443163;
    border-radius: 14px;
    padding: 12px;
}
hr { border-color: #443163; }
</style>
""", unsafe_allow_html=True)


# -------------------- DATA --------------------

@st.cache_data(ttl=4, show_spinner=False)
def get_quote():
    response = requests.get(
        f"{BASE_URL}/EURUSD",
        params={"allowStale": "false"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("symbol") != "EURUSD":
        raise ValueError("La fuente no devolvió EURUSD.")

    if data.get("mid") is None:
        raise ValueError("No hay precio medio disponible.")

    return data


@st.cache_data(ttl=20, show_spinner=False)
def get_candles(interval, limit=200):
    response = requests.get(
        f"{BASE_URL}/EURUSD/ohlc",
        params={"interval": interval, "limit": limit},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    bars = data.get("bars", [])
    if not bars:
        raise ValueError(f"No hay velas disponibles para {interval}.")

    df = pd.DataFrame(bars)

    required = {"openTime", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("La respuesta de velas tiene un formato inesperado.")

    df["openTime"] = pd.to_datetime(df["openTime"], utc=True)
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["openTime", "open", "high", "low", "close"])
        .sort_values("openTime")
        .drop_duplicates(subset=["openTime"])
        .reset_index(drop=True)
    )

    if "isOpen" not in df.columns:
        df["isOpen"] = False

    return df


def add_indicators(df):
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    change = df["close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()
    return df


def analyze(minute_df, quarter_df, minute_in_cycle):
    """
    Technical score, NOT a calibrated win probability.
    Uses closed candles to avoid acting on incomplete bars.
    """
    m = minute_df[~minute_df["isOpen"]].copy()
    q = quarter_df[~quarter_df["isOpen"]].copy()

    if len(m) < 30 or len(q) < 3:
        return {
            "signal": "NO OPERAR",
            "score": 0,
            "reason": "Esperando suficientes velas cerradas.",
            "target": None,
            "atr": None,
        }

    m = add_indicators(m)
    q = add_indicators(q)

    last = m.iloc[-1]
    previous = m.iloc[-2]
    last_q = q.iloc[-1]

    score_up = 0
    score_down = 0
    reasons_up = []
    reasons_down = []

    # Short-term trend
    if last["ema9"] > last["ema21"]:
        score_up += 30
        reasons_up.append("EMA 9 por encima de EMA 21")
    elif last["ema9"] < last["ema21"]:
        score_down += 30
        reasons_down.append("EMA 9 por debajo de EMA 21")

    # Momentum
    if last["close"] > previous["close"]:
        score_up += 15
        reasons_up.append("Última vela cerrada positiva")
    elif last["close"] < previous["close"]:
        score_down += 15
        reasons_down.append("Última vela cerrada negativa")

    # RSI confirmation
    if 52 <= last["rsi"] <= 68:
        score_up += 20
        reasons_up.append("RSI favorece el impulso alcista")
    elif 32 <= last["rsi"] <= 48:
        score_down += 20
        reasons_down.append("RSI favorece el impulso bajista")

    # Higher timeframe confirmation
    if last_q["ema9"] > last_q["ema21"]:
        score_up += 25
        reasons_up.append("Tendencia de 15 minutos alcista")
    elif last_q["ema9"] < last_q["ema21"]:
        score_down += 25
        reasons_down.append("Tendencia de 15 minutos bajista")

    # Avoid signals in extreme RSI conditions
    if last["rsi"] >= 75:
        score_up = max(0, score_up - 25)
        reasons_up.append("Penalización por RSI muy alto")
    if last["rsi"] <= 25:
        score_down = max(0, score_down - 25)
        reasons_down.append("Penalización por RSI muy bajo")

    score_up = min(100, score_up)
    score_down = min(100, score_down)

    # Only evaluate entries from minute 9 onward.
    if minute_in_cycle < 9:
        return {
            "signal": "ESPERANDO MINUTO 9",
            "score": max(score_up, score_down),
            "reason": "El sistema espera a la ventana de análisis.",
            "target": None,
            "atr": last["atr"],
        }

    # Require a strong score and a clear difference between directions.
    if score_up >= 70 and score_up - score_down >= 20:
        signal = "SUBE"
        score = score_up
        reason = " • ".join(reasons_up)
        target = last["close"] + 1.2 * last["atr"] if pd.notna(last["atr"]) else None
    elif score_down >= 70 and score_down - score_up >= 20:
        signal = "BAJA"
        score = score_down
        reason = " • ".join(reasons_down)
        target = last["close"] - 1.2 * last["atr"] if pd.notna(last["atr"]) else None
    else:
        signal = "NO OPERAR"
        score = max(score_up, score_down)
        reason = "Las señales no coinciden con suficiente claridad."
        target = None

    return {
        "signal": signal,
        "score": score,
        "reason": reason,
        "target": target,
        "atr": last["atr"],
    }


def make_chart(df):
    chart_df = df.tail(100)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=chart_df["openTime"],
                open=chart_df["open"],
                high=chart_df["high"],
                low=chart_df["low"],
                close=chart_df["close"],
                name="EUR/USD",
                increasing_line_color="#22d3a6",
                decreasing_line_color="#ff526b",
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="#17102f",
        plot_bgcolor="#17102f",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        xaxis_title="Hora UTC",
        yaxis_title="Precio EUR/USD",
    )
    fig.update_xaxes(gridcolor="#302747")
    fig.update_yaxes(gridcolor="#302747")
    return fig


# -------------------- DASHBOARD --------------------

st.markdown('<div class="eyebrow">LIVE MARKET INTELLIGENCE</div>',
            unsafe_allow_html=True)
st.title("EUR/USD Signal Desk")
st.caption("Analizador técnico de 15 minutos • Solo señales, sin órdenes automáticas")

try:
    quote = get_quote()
    candles_1m = get_candles("1m", 200)
    candles_15m = get_candles("15m", 100)

    timestamp = pd.to_datetime(quote["timestamp"], utc=True)
    age = int(quote.get("quoteAgeSeconds", 999999))
    stale = bool(quote.get("stale", True))
    market_state = str(quote.get("marketState", "unknown")).lower()

    now = datetime.now(timezone.utc)
    seconds_into_cycle = (now.minute % 15) * 60 + now.second
    seconds_left = 15 * 60 - seconds_into_cycle
    minute_in_cycle = seconds_into_cycle // 60

    minutes_left = seconds_left // 60
    seconds_remainder = seconds_left % 60
    timer_text = f"{minutes_left:02d}:{seconds_remainder:02d}"

    price = float(quote["mid"])

    st.markdown(f"""
    <div class="hero">
      <div class="muted">CUENTA REGRESIVA DEL INTERVALO DE 15 MINUTOS</div>
      <div class="timer">{timer_text}</div>
      <div class="muted">Tiempo restante hasta el siguiente intervalo • UTC</div>
      <hr style="border-color:#443163">
      <div class="muted">EUR/USD • PRECIO MEDIO EN VIVO</div>
      <div class="price">{price:.5f}</div>
      <div class="muted">Bid: {float(quote['bid']):.5f} &nbsp; | &nbsp;
      Ask: {float(quote['ask']):.5f}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dirección de la fuente", str(quote.get("direction", "N/D")))
    with c2:
        st.metric("Antigüedad de cotización", f"{age} s")
    with c3:
        st.metric("Ventana actual", f"Minuto {minute_in_cycle:02d} de 15")

    if stale or age > 60 or market_state != "open":
        st.warning(
            "La cotización no se considera suficientemente reciente o "
            "el mercado aparece cerrado. No confíes en señales hasta que "
            "se restablezcan datos frescos."
        )

    analysis = analyze(candles_1m, candles_15m, minute_in_cycle)

    signal = analysis["signal"]
    score = analysis["score"]

    if signal == "SUBE":
        st.markdown(f"""
        <div class="signal-up">
          <div class="muted">SEÑAL TÉCNICA</div>
          <div class="signal-title">🟢 SUBE</div>
          <div>Puntuación técnica: {score}/100</div>
          <div class="muted">No es una probabilidad de ganar.</div>
        </div>
        """, unsafe_allow_html=True)
    elif signal == "BAJA":
        st.markdown(f"""
        <div class="signal-down">
          <div class="muted">SEÑAL TÉCNICA</div>
          <div class="signal-title">🔴 BAJA</div>
          <div>Puntuación técnica: {score}/100</div>
          <div class="muted">No es una probabilidad de ganar.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="signal-wait">
          <div class="muted">ESTADO DEL MERCADO</div>
          <div class="signal-title">⚪ {signal}</div>
          <div>Puntuación técnica: {score}/100</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Análisis del sistema")
    st.write(analysis["reason"])

    if analysis["target"] is not None:
        st.metric(
            "Objetivo técnico orientativo",
            f"{analysis['target']:.5f}",
            help=(
                "Objetivo calculado a partir del ATR de las velas. "
                "No es un take profit garantizado ni un precio de contrato Kalshi."
            ),
        )
    else:
        st.info("No hay objetivo técnico confirmado para esta lectura.")

    st.subheader("Gráfico real de EUR/USD — velas de 1 minuto")
    st.plotly_chart(make_chart(candles_1m), use_container_width=True)

    with st.expander("Ver datos de 15 minutos"):
        st.dataframe(
            candles_15m.tail(20)[
                ["openTime", "open", "high", "low", "close", "isOpen"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"Fuente: BiQuote • {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} • "
        f"Actualización de pantalla cada 5 segundos."
    )

except requests.RequestException as exc:
    st.error(f"No se pudo conectar a BiQuote: {exc}")
    st.info("Comprueba tu conexión y vuelve a abrir la aplicación.")
except (ValueError, KeyError, TypeError, IndexError) as exc:
    st.error(f"No se pudieron procesar los datos: {exc}")
    st.info("La fuente puede estar temporalmente indisponible o haber cambiado.")
except Exception as exc:
    st.error(f"Error inesperado: {exc}")
    st.info("Revisa el registro de errores de Streamlit si persiste.")
