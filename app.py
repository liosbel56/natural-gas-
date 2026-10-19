
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="EUR/USD Signal Desk",
    page_icon="📈",
    layout="wide"
)

API = "https://biquote.io/api"
TIMEOUT = 12

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg,#10091f,#20113b,#101426);
    color: white;
}
.block-container {max-width:1200px;padding-top:1.5rem}
.hero,.signal {
    background:#211536;
    border:1px solid #7146a8;
    border-radius:18px;
    padding:20px;
    margin:12px 0;
}
.timer {color:#ff334f;font-size:64px;font-weight:900}
.price {font-size:42px;font-weight:850;color:white}
.muted {color:#c1b5d7;font-size:13px}
</style>
""", unsafe_allow_html=True)


def get_json(url, params=None):
    response = requests.get(
        url, params=params, timeout=TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Formato de datos inesperado.")
    return data


def quote_data():
    data = get_json(f"{API}/EURUSD")
    if data.get("symbol") != "EURUSD" or data.get("mid") is None:
        raise ValueError("Cotización EUR/USD no válida.")
    return data


def candle_data(interval):
    data = get_json(
        f"{API}/EURUSD/ohlc",
        {"interval": interval, "limit": 200}
    )

    bars = data.get("bars", [])
    if not bars:
        raise ValueError(f"No hay velas para {interval}.")

    df = pd.DataFrame(bars)
    required = ["openTime", "open", "high", "low", "close"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta el campo {col}.")

    df["openTime"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "isOpen" not in df.columns:
        df["isOpen"] = False
    else:
        df["isOpen"] = (
            df["isOpen"].astype(str).str.lower()
            .isin(["true", "1", "yes"])
        )

    return (
        df.dropna(subset=required)
        .sort_values("openTime")
        .drop_duplicates("openTime")
        .reset_index(drop=True)
    )


def add_indicators(df):
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA 9 y EMA 21
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()

    # RSI 14
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "rsi"] = 0

    # MACD 12, 26, 9
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands 20, 2
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_mid"] = middle
    df["bb_upper"] = middle + 2 * std
    df["bb_lower"] = middle - 2 * std

    # ADX 14 y líneas direccionales
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    df["plus_di"] = 100 * plus_smoothed / atr.replace(
        0, float("nan")
    )
    df["minus_di"] = 100 * minus_smoothed / atr.replace(
        0, float("nan")
    )

    di_sum = (df["plus_di"] + df["minus_di"]).replace(
        0, float("nan")
    )

    dx = 100 * (
        df["plus_di"] - df["minus_di"]
    ).abs() / di_sum

    df["adx"] = dx.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    return df


def analyze(one_min, fifteen_min):
    m = add_indicators(
        one_min[~one_min["isOpen"]].copy()
    ).reset_index(drop=True)

    q = add_indicators(
        fifteen_min[~fifteen_min["isOpen"]].copy()
    ).reset_index(drop=True)

    if len(m) < 60 or len(q) < 60:
        return (
            "WAIT", 0, 0,
            "Faltan velas cerradas para calcular los indicadores."
        )

    x = m.iloc[-1]
    trend = q.iloc[-1]

    fields = [
        "ema9", "ema21", "rsi", "macd",
        "macd_signal", "macd_hist", "bb_mid",
        "bb_upper", "bb_lower", "adx",
        "plus_di", "minus_di"
    ]

    if any(pd.isna(x[f]) for f in fields):
        return "WAIT", 0, 0, "Esperando indicadores válidos."

    if any(pd.isna(trend[f]) for f in ["ema9", "ema21"]):
        return "WAIT", 0, 0, "Esperando tendencia de 15 minutos."

    up = 0
    down = 0
    reasons = []

    # EMA: 20 puntos
    if x["ema9"] > x["ema21"]:
        up += 20
        reasons.append("EMA 1m alcista")
    elif x["ema9"] < x["ema21"]:
        down += 20
        reasons.append("EMA 1m bajista")

    # RSI: 15 puntos
    if 52 <= x["rsi"] <= 68:
        up += 15
        reasons.append("RSI alcista")
    elif 32 <= x["rsi"] <= 48:
        down += 15
        reasons.append("RSI bajista")

    # MACD: 20 puntos
    if x["macd"] > x["macd_signal"] and x["macd_hist"] > 0:
        up += 20
        reasons.append("MACD alcista")
    elif x["macd"] < x["macd_signal"] and x["macd_hist"] < 0:
        down += 20
        reasons.append("MACD bajista")

    # Bollinger: 10 puntos
    if x["close"] > x["bb_mid"] and x["close"] < x["bb_upper"]:
        up += 10
        reasons.append("Precio sobre banda media")
    elif x["close"] < x["bb_mid"] and x["close"] > x["bb_lower"]:
        down += 10
        reasons.append("Precio bajo banda media")

    # ADX y DI: 15 puntos
    if x["adx"] >= 20:
        if x["plus_di"] > x["minus_di"]:
            up += 15
            reasons.append("ADX/DI alcista")
        elif x["minus_di"] > x["plus_di"]:
            down += 15
            reasons.append("ADX/DI bajista")

    # Confirmación en velas de 15 minutos: 20 puntos
    if trend["ema9"] > trend["ema21"]:
        up += 20
        reasons.append("Tendencia 15m alcista")
    elif trend["ema9"] < trend["ema21"]:
        down += 20
        reasons.append("Tendencia 15m bajista")

    if up >= 60 and up - down >= 20:
        return "BUY UP", up, down, "; ".join(reasons)

    if down >= 60 and down - up >= 20:
        return "BUY DOWN", up, down, "; ".join(reasons)

    return (
        "WAIT", up, down,
        "Indicadores sin suficiente confirmación. "
        + "; ".join(reasons)
    )


def show_dashboard():
    try:
        quote = quote_data()
        one = candle_data("1m")
        fifteen = candle_data("15m")

        now = datetime.now(timezone.utc)
        seconds_used = (now.minute % 15) * 60 + now.second
        remaining = 900 - seconds_used
        timer = f"{remaining // 60:02d}:{remaining % 60:02d}"

        price = float(quote["mid"])
        age = int(quote.get("quoteAgeSeconds", 9999))
        stale = bool(quote.get("stale", True))

        st.markdown(
            '<div class="muted">LIVE MARKET INTELLIGENCE</div>'
            '<h1>EUR/USD SIGNAL DESK</h1>'
            '<p>Indicadores técnicos • Horizonte de 15 minutos</p>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="hero">
            <div class="muted">CUENTA REGRESIVA • 15 MINUTOS</div>
            <div class="timer">{timer}</div>
            <div class="muted">Referencia UTC</div>
            <hr>
            <div class="muted">PRECIO MEDIO EUR/USD</div>
            <div class="price">{price:.5f}</div>
            <p>BID: {float(quote.get('bid', price)):.5f}
            &nbsp; | &nbsp;
            ASK: {float(quote.get('ask', price)):.5f}</p>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        a.metric("Dirección de fuente", str(quote.get("direction", "N/D")))
        b.metric("Antigüedad del precio", f"{age} segundos")
        c.metric("Velas de 1m", len(one[~one["isOpen"]]))

        if stale or age > 60:
            st.warning("Cotización atrasada: señales desactivadas.")

        signal, up, down, reason = analyze(one, fifteen)

        if stale or age > 60:
            signal = "WAIT"
            reason = "Los datos de cotización están atrasados."

        if signal == "BUY UP":
            st.success("🟢 BUY UP — sesgo alcista")
        elif signal == "BUY DOWN":
            st.error("🔴 BUY DOWN — sesgo bajista")
        else:
            st.warning("🟡 WAIT — sin confirmación suficiente")

        left, right = st.columns(2)
        left.metric("Puntuación alcista", f"{up}/100")
        right.metric("Puntuación bajista", f"{down}/100")

        st.markdown(
            f'<div class="signal"><b>Análisis de indicadores</b>'
            f'<p>{reason}</p></div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Las puntuaciones son reglas técnicas, no probabilidades "
            "de ganar. Las señales no garantizan resultados."
        )

        chart = go.Figure(data=[go.Candlestick(
            x=one["openTime"],
            open=one["open"],
            high=one["high"],
            low=one["low"],
            close=one["close"],
            increasing_line_color="#27dda7",
            decreasing_line_color="#ff526b",
            name="EUR/USD"
        )])

        chart.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#17102b",
            plot_bgcolor="#17102b",
            margin=dict(l=10, r=10, t=25, b=10)
        )

        st.subheader("Gráfico EUR/USD de 1 minuto")
        st.plotly_chart(chart, use_container_width=True)

        tech = add_indicators(
            one[~one["isOpen"]].copy()
        )

        if not tech.empty:
            last = tech.iloc[-1]
            with st.expander("Ver indicadores técnicos"):
                st.dataframe(pd.DataFrame({
                    "Indicador": [
                        "EMA 9", "EMA 21", "RSI (14)",
                        "MACD", "MACD Signal", "MACD Histogram",
                        "Bollinger superior", "Bollinger media",
                        "Bollinger inferior", "ADX (14)", "+DI", "-DI"
                    ],
                    "Valor": [
                        last["ema9"], last["ema21"], last["rsi"],
                        last["macd"], last["macd_signal"],
                        last["macd_hist"], last["bb_upper"],
                        last["bb_mid"], last["bb_lower"],
                        last["adx"], last["plus_di"], last["minus_di"]
                    ]
                }).round(5), use_container_width=True, hide_index=True)

        with st.expander("Ver últimas velas de 15 minutos"):
            st.dataframe(
                fifteen.tail(20),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            f"Fuente: BiQuote • {quote.get('timestamp', 'Hora no disponible')}"
        )

    except requests.RequestException as exc:
        st.error("No se pudo conectar con BiQuote.")
        st.code(str(exc))
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        st.error("No se pudieron procesar los datos.")
        st.code(str(exc))
    except Exception as exc:
        st.error("Error inesperado.")
        st.code(str(exc))


@st.fragment(run_every="5s")
def live_panel():
    show_dashboard()


live_panel()
