
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
h1,h2,h3,p,label {color:#f5f0ff}
.hero {
    background:#211536;border:1px solid #7146a8;
    border-radius:22px;padding:24px;margin-bottom:16px;
}
.timer {
    color:#ff334f;font-size:clamp(55px,10vw,88px);
    font-weight:900;line-height:1.15;
}
.price {font-size:48px;font-weight:850;color:white}
.muted {color:#c1b5d7;font-size:13px}
.signal {
    background:#25193d;border:1px solid #8c67bd;
    border-radius:18px;padding:20px;margin:12px 0;
}
</style>
""", unsafe_allow_html=True)


def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("La fuente devolvió un formato inesperado.")
    return data


def quote_data():
    data = get_json(f"{API}/EURUSD")
    if data.get("symbol") != "EURUSD" or data.get("mid") is None:
        raise ValueError("No llegó una cotización válida de EUR/USD.")
    return data


def candle_data(interval):
    data = get_json(
        f"{API}/EURUSD/ohlc",
        {"interval": interval, "limit": 200}
    )
    bars = data.get("bars", [])
    if not bars:
        raise ValueError(f"No hay velas disponibles para {interval}.")

    df = pd.DataFrame(bars)
    required = ["openTime", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta el campo {col} en las velas.")

    df["openTime"] = pd.to_datetime(df["openTime"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "isOpen" not in df.columns:
        df["isOpen"] = False

    return (
        df.dropna(subset=required)
        .sort_values("openTime")
        .drop_duplicates("openTime")
        .reset_index(drop=True)
    )


def indicators(df):
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    change = df["close"].diff()
    gain = change.clip(lower=0).ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()
    loss = (-change.clip(upper=0)).ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    return df


def analyze(one_min, fifteen_min, minute):
    m = indicators(one_min[~one_min["isOpen"]].copy())
    q = indicators(fifteen_min[~fifteen_min["isOpen"]].copy())

    if len(m) < 30 or len(q) < 3:
        return "NO OPERAR", 0, "Esperando suficientes velas cerradas."

    last = m.iloc[-1]
    prev = m.iloc[-2]
    last15 = q.iloc[-1]

    up = 0
    down = 0
    reasons = []

    if last["ema9"] > last["ema21"]:
        up += 30
        reasons.append("Tendencia corta alcista")
    elif last["ema9"] < last["ema21"]:
        down += 30
        reasons.append("Tendencia corta bajista")

    if last["close"] > prev["close"]:
        up += 15
    elif last["close"] < prev["close"]:
        down += 15

    if 52 <= last["rsi"] <= 68:
        up += 20
    elif 32 <= last["rsi"] <= 48:
        down += 20

    if last15["ema9"] > last15["ema21"]:
        up += 25
    elif last15["ema9"] < last15["ema21"]:
        down += 25

    if last["rsi"] > 75:
        up = max(0, up - 20)
    if last["rsi"] < 25:
        down = max(0, down - 20)

    if minute < 9:
        return "ESPERANDO MINUTO 9", max(up, down), \
            "El sistema espera la ventana de análisis."

    if up >= 70 and up - down >= 20:
        return "SUBE", up, "Los indicadores favorecen el impulso alcista."
    if down >= 70 and down - up >= 20:
        return "BAJA", down, "Los indicadores favorecen el impulso bajista."

    return "NO OPERAR", max(up, down), \
        "No hay suficiente confirmación entre los indicadores."


def show_dashboard():
    try:
        quote = quote_data()
        one = candle_data("1m")
        fifteen = candle_data("15m")

        now = datetime.now(timezone.utc)
        seconds_used = (now.minute % 15) * 60 + now.second
        remaining = 900 - seconds_used
        timer = f"{remaining // 60:02d}:{remaining % 60:02d}"
        minute = seconds_used // 60

        price = float(quote["mid"])
        age = int(quote.get("quoteAgeSeconds", 9999))
        stale = bool(quote.get("stale", True))

        st.markdown(
            '<div class="muted">LIVE MARKET INTELLIGENCE</div>'
            '<h1>EUR/USD Signal Desk</h1>'
            '<div class="muted">Analizador técnico • Solo señales</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="hero">
            <div class="muted">CUENTA REGRESIVA • INTERVALO 15 MIN</div>
            <div class="timer">{timer}</div>
            <div class="muted">Hora de referencia: UTC</div>
            <hr>
            <div class="muted">PRECIO MEDIO EUR/USD</div>
            <div class="price">{price:.5f}</div>
            <div class="muted">
                BID: {float(quote.get('bid', price)):.5f}
                &nbsp; | &nbsp;
                ASK: {float(quote.get('ask', price)):.5f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        a.metric("Dirección de la fuente", str(quote.get("direction", "N/D")))
        b.metric("Antigüedad del precio", f"{age} segundos")
        c.metric("Minuto del intervalo", f"{minute} / 15")

        if stale or age > 60:
            st.warning("Precio atrasado. No confíes en las señales actuales.")

        signal, score, reason = analyze(one, fifteen, minute)

        if stale or age > 60:
            signal = "NO OPERAR"
            reason = "Datos atrasados: señal desactivada."

        st.markdown(
            f'<div class="signal"><div class="muted">SEÑAL TÉCNICA</div>'
            f'<h2>{signal}</h2>'
            f'<p>Puntuación técnica: {score}/100</p>'
            f'<p>{reason}</p></div>',
            unsafe_allow_html=True
        )

        st.caption(
            "La puntuación no es una probabilidad de ganar. "
            "No garantiza un 80% de acierto ni ejecuta operaciones."
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

        st.subheader("Gráfico de velas de 1 minuto")
        st.plotly_chart(chart, use_container_width=True)

        with st.expander("Ver velas de 15 minutos"):
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
        st.error("Los datos recibidos no tienen el formato esperado.")
        st.code(str(exc))
    except Exception as exc:
        st.error("Se produjo un error inesperado.")
        st.code(str(exc))


@st.fragment(run_every="5s")
def live_panel():
    show_dashboard()


live_panel()
