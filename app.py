
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="BTC Signals | Kalshi",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BTC_URL = (
    "https://api.exchange.coinbase.com"
    "/products/BTC-USD/candles?granularity=60"
)

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"

# Umbral técnico: no representa una probabilidad de ganar.
UMBRAL_SEÑAL = 4

# Actualiza la pantalla cada 5 segundos.
# Las velas de Coinbase cambian aproximadamente cada minuto.
st_autorefresh(interval=5000, key="btc_refresh")


# =========================================================
# TEMA MORADO
# =========================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            #462078 0%,
            #24103d 40%,
            #10091d 100%
        );
    color: #f7f0ff;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 1.4rem;
    max-width: 1250px;
}
div[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(70, 35, 105, 0.30);
    border: 1px solid #654095;
    border-radius: 15px;
    padding: 12px;
}
h1, h2, h3 {
    color: #f4eaff !important;
}
.signal-card {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, #321650, #201034);
    border: 1px solid #8952c7;
    margin: 10px 0 18px 0;
}
.signal-up {
    color: #4af0a5;
    font-size: 25px;
    font-weight: 800;
}
.signal-down {
    color: #ff718d;
    font-size: 25px;
    font-weight: 800;
}
.signal-wait {
    color: #f5d27c;
    font-size: 25px;
    font-weight: 800;
}
.small-note {
    color: #c7b4e3;
    font-size: 12px;
}
.stButton > button {
    background: linear-gradient(135deg, #a855f7, #7138ca);
    color: white;
    border: 1px solid #c084fc;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATOS BTC/USD
# =========================================================

@st.cache_data(ttl=20, show_spinner=False)
def obtener_velas_btc():
    respuesta = requests.get(
        BTC_URL,
        timeout=10,
        headers={"User-Agent": "BTC-Signals-Dashboard"},
    )
    respuesta.raise_for_status()

    datos = respuesta.json()

    if not isinstance(datos, list) or len(datos) < 60:
        raise ValueError("Coinbase no devolvió suficientes velas.")

    df = pd.DataFrame(
        datos,
        columns=[
            "time", "low", "high", "open", "close", "volume"
        ],
    )

    df = df.sort_values("time").reset_index(drop=True)

    for col in ["low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )

    return df.dropna().reset_index(drop=True)


# =========================================================
# INDICADORES TÉCNICOS
# =========================================================

def calcular_indicadores(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    # RSI 14
    cambio = df["close"].diff()
    ganancias = cambio.clip(lower=0)
    perdidas = -cambio.clip(upper=0)

    media_g = ganancias.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    media_p = perdidas.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = media_g / media_p.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # MACD
    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # ADX aproximado
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(
        (plus_dm > minus_dm) & (plus_dm > 0), 0
    )

    minus_dm = minus_dm.where(
        (minus_dm > plus_dm) & (minus_dm > 0), 0
    )

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    plus_di = (
        100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["adx"] = dx.ewm(
        alpha=1 / 14, adjust=False
    ).mean().fillna(0)

    # Bandas de Bollinger
    media20 = close.rolling(20).mean()
    desviacion20 = close.rolling(20).std()

    df["bb_upper"] = media20 + 2 * desviacion20
    df["bb_lower"] = media20 - 2 * desviacion20
    df["bb_mid"] = media20

    # Movimiento de precio en 3 y 5 minutos
    df["change3"] = close.pct_change(3) * 100
    df["change5"] = close.pct_change(5) * 100

    # Volumen relativo
    df["volume_avg"] = df["volume"].rolling(20).mean()

    return df


# =========================================================
# MOTOR DE SEÑALES BTC
# =========================================================

def analizar_btc(df):
    df = calcular_indicadores(df)

    actual = df.iloc[-1]
    anterior = df.iloc[-2]

    puntuacion = 0
    motivos = []
    alertas = []

    # 1. Tendencia de corto plazo
    if actual["ema9"] > actual["ema21"]:
        puntuacion += 1
        motivos.append("EMA 9 por encima de EMA 21")
    else:
        puntuacion -= 1
        motivos.append("EMA 9 por debajo de EMA 21")

    # 2. Tendencia general
    if actual["ema21"] > actual["ema50"]:
        puntuacion += 1
        motivos.append("Tendencia general alcista")
    elif actual["ema21"] < actual["ema50"]:
        puntuacion -= 1
        motivos.append("Tendencia general bajista")

    # 3. Impulso MACD
    if actual["macd"] > actual["macd_signal"]:
        puntuacion += 1
        motivos.append("MACD con impulso alcista")
    else:
        puntuacion -= 1
        motivos.append("MACD con impulso bajista")

    # 4. RSI
    if 52 <= actual["rsi"] <= 68:
        puntuacion += 1
        motivos.append("RSI favorece impulso alcista")
    elif 32 <= actual["rsi"] <= 48:
        puntuacion -= 1
        motivos.append("RSI favorece impulso bajista")
    elif actual["rsi"] >= 75:
        alertas.append("RSI elevado: posible sobrecompra")
    elif actual["rsi"] <= 25:
        alertas.append("RSI bajo: posible sobreventa")

    # 5. Movimiento reciente
    cambio3 = float(
        actual["change3"]
        if pd.notna(actual["change3"]) else 0
    )

    if cambio3 > 0.05:
        puntuacion += 1
        motivos.append("Precio subiendo en los últimos 3 minutos")
    elif cambio3 < -0.05:
        puntuacion -= 1
        motivos.append("Precio bajando en los últimos 3 minutos")

    # 6. Cruces de medias para posibles reversiones
    cruce_alcista = (
        anterior["ema9"] <= anterior["ema21"]
        and actual["ema9"] > actual["ema21"]
    )

    cruce_bajista = (
        anterior["ema9"] >= anterior["ema21"]
        and actual["ema9"] < actual["ema21"]
    )

    if cruce_alcista:
        alertas.append("Cruce alcista reciente de EMA 9/21")

    if cruce_bajista:
        alertas.append("Cruce bajista reciente de EMA 9/21")

    # 7. Fuerza de tendencia
    if actual["adx"] >= 20:
        motivos.append("ADX indica tendencia con fuerza")
    else:
        motivos.append("ADX bajo: tendencia poco definida")

    # 8. Evitar señales de entrada en extremos
    if puntuacion >= UMBRAL_SEÑAL and actual["rsi"] < 72:
        señal = "COMPRAR UP — CONFIRMACIÓN TÉCNICA"
        tipo = "up"

    elif puntuacion <= -UMBRAL_SEÑAL and actual["rsi"] > 28:
        señal = "COMPRAR DOWN — CONFIRMACIÓN TÉCNICA"
        tipo = "down"

    elif cruce_bajista:
        señal = "POSIBLE CAÍDA — REVISAR"
        tipo = "down"

    elif cruce_alcista:
        señal = "POSIBLE SUBIDA — REVISAR"
        tipo = "up"

    elif puntuacion >= 2:
        señal = "POSIBLE SUBIDA — ESPERAR CONFIRMACIÓN"
        tipo = "wait"

    elif puntuacion <= -2:
        señal = "POSIBLE CAÍDA — ESPERAR CONFIRMACIÓN"
        tipo = "wait"

    else:
        señal = "ESPERAR — SEÑALES MIXTAS"
        tipo = "wait"

    # No dar una entrada nueva si la vela más reciente está vieja.
    segundos_desde_vela = (
        datetime.now(timezone.utc)
        - actual["time"].to_pydatetime()
    ).total_seconds()

    if segundos_desde_vela > 180:
        señal = "DATOS ANTIGUOS — NO ENTRAR"
        tipo = "wait"
        alertas.append("La última vela tiene más de 3 minutos")

    return {
        "df": df,
        "precio": float(actual["close"]),
        "rsi": float(actual["rsi"]),
        "macd": float(actual["macd"]),
        "adx": float(actual["adx"]),
        "ema9": float(actual["ema9"]),
        "ema21": float(actual["ema21"]),
        "ema50": float(actual["ema50"]),
        "puntuacion": puntuacion,
        "cambio3": cambio3,
        "señal": señal,
        "tipo": tipo,
        "motivos": motivos,
        "alertas": alertas,
        "hora_vela": actual["time"],
    }


# =========================================================
# CONSULTA INFORMATIVA DE KALSHI
# =========================================================

@st.cache_data(ttl=15, show_spinner=False)
def obtener_mercados_btc_kalshi():
    """
    Intenta consultar mercados abiertos de la serie BTC de 15 min.
    La serie puede cambiar o no estar disponible en esta API.
    No se inventan cotizaciones si la consulta falla.
    """
    url = f"{KALSHI_API}/markets"

    respuesta = requests.get(
        url,
        params={
            "series_ticker": "KXBTC15M",
            "status": "open",
            "limit": 10,
        },
        timeout=8,
    )

    respuesta.raise_for_status()
    datos = respuesta.json()

    return datos.get("markets", [])


# =========================================================
# INTERFAZ
# =========================================================

st.title("₿ BTC/USD — Trading Signals")
st.caption(
    "Análisis técnico de 1 minuto para contratos de 15 minutos. "
    "Las señales son estimaciones, no garantías."
)

st_autorefresh(interval=5000, key="refresh_timer")

try:
    velas = obtener_velas_btc()
    resultado = analizar_btc(velas)

    precio = resultado["precio"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Precio BTC/USD",
        f"${precio:,.2f}",
        f"{resultado['cambio3']:+.3f}% en 3 min",
    )

    col2.metric(
        "RSI (14)",
        f"{resultado['rsi']:.1f}",
    )

    col3.metric(
        "Puntuación técnica",
        f"{resultado['puntuacion']:+d}",
        "No es porcentaje de acierto",
    )

    clase = (
        "signal-up" if resultado["tipo"] == "up"
        else "signal-down" if resultado["tipo"] == "down"
        else "signal-wait"
    )

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="{clase}">{resultado['señal']}</div>
            <p class="small-note">
                Última vela: {resultado['hora_vela'].strftime('%H:%M:%S UTC')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gráfico con velas y EMA
    grafico_df = resultado["df"].tail(90)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=grafico_df["time"],
            open=grafico_df["open"],
            high=grafico_df["high"],
            low=grafico_df["low"],
            close=grafico_df["close"],
            name="BTC/USD",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grafico_df["time"],
            y=grafico_df["ema9"],
            mode="lines",
            name="EMA 9",
            line=dict(color="#60a5fa", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grafico_df["time"],
            y=grafico_df["ema21"],
            mode="lines",
            name="EMA 21",
            line=dict(color="#fbbf24", width=2),
        )
    )

    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,8,38,0.7)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Indicadores")
        st.write(f"EMA 9: ${resultado['ema9']:,.2f}")
        st.write(f"EMA 21: ${resultado['ema21']:,.2f}")
        st.write(f"EMA 50: ${resultado['ema50']:,.2f}")
        st.write(f"MACD: {resultado['macd']:.4f}")
        st.write(f"ADX: {resultado['adx']:.1f}")

        with st.expander("Motivos de la señal"):
            for motivo in resultado["motivos"]:
                st.write(f"• {motivo}")

    with derecha:
        st.subheader("Alertas de reversión")

        if resultado["alertas"]:
            for alerta in resultado["alertas"]:
                st.warning(alerta)
        else:
            st.write("No se detectó un cruce nuevo de EMA 9/21.")

        st.caption(
            "Los cruces pueden llegar después del comienzo de un movimiento. "
            "No garantizan una reversión."
        )

    st.subheader("Mercados de BTC en Kalshi")

    if st.button("Actualizar mercados de Kalshi"):
        obtener_mercados_btc_kalshi.clear()

    try:
        mercados = obtener_mercados_btc_kalshi()

        if mercados:
            for mercado in mercados:
                ticker = mercado.get("ticker", "Sin ticker")
                titulo = mercado.get(
                    "title", mercado.get("subtitle", ticker)
                )

                st.write(f"**{titulo}**")
                st.caption(f"Contrato: {ticker}")

                # Muestra los campos publicados por la API.
                # No supone que un campo equivalga a probabilidad
                # de liquidación sin revisar las reglas del contrato.
                campos = [
                    "yes_bid_dollars",
                    "yes_ask_dollars",
                    "no_bid_dollars",
                    "no_ask_dollars",
                    "close_time",
                    "status",
                ]

                datos_visibles = {
                    campo: mercado[campo]
                    for campo in campos
                    if campo in mercado
                }

                if datos_visibles:
                    st.json(datos_visibles)

                st.divider()
        else:
            st.info(
                "La consulta no devolvió mercados abiertos de esa serie. "
                "Verifica el ticker y las reglas del contrato en Kalshi."
            )

    except Exception:
        st.info(
            "No se pudieron cargar los mercados de Kalshi. "
            "Las señales técnicas siguen basándose en Coinbase, "
            "no en la cotización del contrato."
        )

except Exception as error:
    st.error(f"No se pudo analizar BTC/USD: {error}")
    st.info(
        "Revisa la conexión y vuelve a intentarlo. "
        "No uses señales anteriores como si fueran actuales."
    )

st.divider()

st.caption(
    "Importante: este programa no ejecuta compras. "
    "Coinbase es una referencia de precio, no necesariamente "
    "la fuente de liquidación de Kalshi. La señal técnica no "
    "es una probabilidad validada de ganar."
)
