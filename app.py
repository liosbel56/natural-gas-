
import re
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIGURACION
# =====================================================
st.set_page_config(
    page_title="Purple Signals AI",
    page_icon="💜",
    layout="wide"
)

COINBASE = "https://api.exchange.coinbase.com"
BIQUOTE = "https://biquote.io/api"
KALSHI = "https://external-api.kalshi.com/trade-api/v2"
TIMEOUT = 10

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg,#10091f,#211039,#35165b);
    color: white;
}
h1,h2,h3,p,label { color: white; }
div[data-testid="stMetric"] {
    background: #281640;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #69439a;
}
.signal {
    text-align: center;
    font-size: clamp(40px, 9vw, 68px);
    font-weight: 900;
    padding: 22px;
    border-radius: 18px;
    margin: 14px 0;
}
.up {
    background: #103d2b;
    color: #4cff9b;
    border: 2px solid #4cff9b;
}
.down {
    background: #491c30;
    color: #ff718f;
    border: 2px solid #ff718f;
}
.wait {
    background: #40321a;
    color: #ffda70;
    border: 2px solid #ffda70;
}
.panel {
    background: #291941;
    border: 1px solid #8058aa;
    border-radius: 12px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

# Recarga de pantalla para actualizar temporizador.
# Las llamadas a las APIs se limitan mediante caché.
st_autorefresh(interval=1000, key="screen_refresh")

st.title("💜 PURPLE SIGNALS AI")
st.caption("BTC/USD + EUR/USD | Predicción Above / Below | Ciclos de 15 minutos")

# =====================================================
# TEMPORIZADOR UTC
# =====================================================
def show_timer():
    now = datetime.now(timezone.utc)
    elapsed = (now.minute % 15) * 60 + now.second
    remaining = 900 - elapsed
    mm, ss = divmod(remaining, 60)

    st.markdown(
        f"""
        <div class="panel" style="text-align:center">
          <div style="font-size:15px;color:#c9b8e8">
            TIEMPO RESTANTE DEL CICLO UTC
          </div>
          <div style="font-size:52px;font-weight:900;color:#d0a8ff">
            {mm:02d}:{ss:02d}
          </div>
          <div style="color:#c9b8e8;font-size:12px">
            Nuevo ciclo al llegar a 00:00
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

show_timer()

# =====================================================
# FUNCIONES GENERALES
# =====================================================
def get_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT,
        headers={"User-Agent": "PurpleSignals/1.0"}
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10)
def get_btc_candles():
    """Velas publicas de Coinbase. Solo velas cerradas."""
    now = int(time.time())
    end = now - (now % 60)
    start = end - 240 * 60

    data = get_json(
        f"{COINBASE}/products/BTC-USD/candles",
        {
            "granularity": 60,
            "start": datetime.fromtimestamp(
                start, timezone.utc
            ).isoformat(),
            "end": datetime.fromtimestamp(
                end, timezone.utc
            ).isoformat()
        }
    )

    df = pd.DataFrame(
        data,
        columns=["time", "low", "high", "open", "close", "volume"]
    )
    if df.empty:
        raise ValueError("Coinbase no devolvio velas BTC/USD.")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[df["time"] < pd.Timestamp(end, unit="s", tz="UTC")]
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data(ttl=10)
def get_eurusd_candles():
    """Velas EUR/USD de BiQuote."""
    data = get_json(
        f"{BIQUOTE}/EURUSD/ohlc",
        {"interval": "1m", "limit": 240}
    )

    df = pd.DataFrame(data.get("bars", []))
    if df.empty or "close" not in df.columns:
        raise ValueError(
            "BiQuote no devolvio velas con el formato esperado."
        )

    df["time"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["time", "close"])

    if "isOpen" in df.columns:
        df = df[~df["isOpen"].astype(bool)]

    return df.sort_values("time").reset_index(drop=True)


@st.cache_data(ttl=10)
def get_btc_price():
    data = get_json(f"{COINBASE}/products/BTC-USD/ticker")
    return float(data["price"])


@st.cache_data(ttl=10)
def get_eurusd_price():
    data = get_json(f"{BIQUOTE}/EURUSD")
    if "mid" not in data:
        raise ValueError("La API no devolvio el precio medio EUR/USD.")
    return float(data["mid"]), data


# =====================================================
# INDICADORES TECNICOS
# =====================================================
def add_indicators(df):
    df = df.copy()
    close = df["close"].astype(float)

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/14, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    # ATR como medida aproximada de volatilidad
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()
    return df


def get_score(df):
    """Puntuacion orientativa; no es una probabilidad."""
    last = df.iloc[-1]
    score = 0

    if last["ema9"] > last["ema21"]:
        score += 1
    else:
        score -= 1

    if last["ema21"] > last["ema50"]:
        score += 1
    else:
        score -= 1

    if last["close"] > last["ema9"]:
        score += 1
    else:
        score -= 1

    if pd.notna(last["rsi"]):
        if 52 <= last["rsi"] <= 68:
            score += 1
        elif 32 <= last["rsi"] <= 48:
            score -= 1

    if last["macd_hist"] > 0:
        score += 1
    else:
        score -= 1

    if len(df) >= 3:
        if last["macd_hist"] > df.iloc[-2]["macd_hist"]:
            score += 1
        elif last["macd_hist"] < df.iloc[-2]["macd_hist"]:
            score -= 1

    return score


# =====================================================
# DATOS PUBLICOS DE KALSHI
# =====================================================
@st.cache_data(ttl=20)
def get_open_markets():
    markets = []
    cursor = None

    # Consultar varias paginas de mercados abiertos.
    for _ in range(5):
        params = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor

        data = get_json(f"{KALSHI}/markets", params)
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")

        if not cursor:
            break

    return markets


def market_text(m):
    return " ".join(
        str(m.get(k, ""))
        for k in ["ticker", "title", "subtitle", "event_ticker"]
    ).lower()


def find_target(m, asset):
    """Busca strike numerico o precio explicito en el titulo."""
    for key in ["floor_strike", "cap_strike", "strike"]:
        value = m.get(key)
        if value is not None:
            try:
                number = float(value)
                if number > 0:
                    return number
            except (ValueError, TypeError):
                pass

    text = " ".join([
        str(m.get("title", "")),
        str(m.get("subtitle", ""))
    ])

    # Primero buscar un precio con simbolo de dolar.
    matches = re.findall(r"\$([0-9,]+(?:\.[0-9]+)?)", text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass

    # Para EUR/USD, buscar un nivel como 1.1650.
    if asset == "EUR/USD":
        matches = re.findall(r"(?<!\d)(1\.\d{3,6})(?!\d)", text)
        if matches:
            return float(matches[-1])

    return None


def get_contract_price_cents(m):
    for key in ["yes_ask_dollars", "yes_bid_dollars"]:
        value = m.get(key)
        if value is not None:
            try:
                return float(value) * 100
            except (ValueError, TypeError):
                pass

    for key in ["yes_ask", "yes_bid"]:
        value = m.get(key)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                pass

    return None


def get_expiration(m):
    for key in [
        "close_time", "expiration_time", "expected_expiration_time"
    ]:
        value = m.get(key)
        if value:
            try:
                return pd.to_datetime(value, utc=True)
            except Exception:
                pass
    return None


# =====================================================
# SELECCION DE MERCADO
# =====================================================
asset = st.selectbox("ELIGE EL ACTIVO", ["BTC/USD", "EUR/USD"])

try:
    if asset == "BTC/USD":
        raw_df = get_btc_candles()
        price = get_btc_price()
        stale = False
        market_terms = ["bitcoin", "btc"]
    else:
        raw_df = get_eurusd_candles()
        price, quote = get_eurusd_price()
        stale = bool(quote.get("stale", False))
        market_terms = ["eur/usd", "eurusd", "euro/usd"]

    if len(raw_df) < 60:
        st.warning("Todavia no hay suficientes velas para analizar.")
        st.stop()

    df = add_indicators(raw_df)
    score = get_score(df)
    last = df.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio actual", f"{price:,.5f}" if asset == "EUR/USD" else f"${price:,.2f}")
    c2.metric("Puntuacion tecnica", f"{score:+d} / 6")
    c3.metric("RSI (14)", f"{last['rsi']:.1f}" if pd.notna(last["rsi"]) else "N/D")

    if stale:
        st.warning("EUR/USD puede estar desactualizado.")

    st.subheader("Contrato de Kalshi")

    try:
        markets = get_open_markets()
        candidates = [
            m for m in markets
            if any(term in market_text(m) for term in market_terms)
        ]
    except Exception as e:
        st.error("No se pudieron consultar los mercados de Kalshi.")
        st.caption(f"{type(e).__name__}: {e}")
        candidates = []

    contract = None
    if candidates:
        options = list(range(len(candidates)))
        selected = st.selectbox(
            "Selecciona el contrato que coincide con tu pantalla",
            options,
            format_func=lambda i: (
                f"{candidates[i].get('title', 'Contrato')} "
                f"[{candidates[i].get('ticker', '')}]"
            )
        )
        contract = candidates[selected]
    else:
        st.warning(
            "No se encontraron contratos abiertos coincidentes. "
            "No se generara una prediccion ligada a Kalshi hasta identificar "
            "el contrato correcto."
        )

    target = find_target(contract, asset) if contract else None
    cents = get_contract_price_cents(contract) if contract else None
    expiry = get_expiration(contract) if contract else None

    if contract:
        a, b, c = st.columns(3)
        a.metric(
            "Objetivo Kalshi",
            f"{target:,.5f}" if target is not None else "No identificado"
        )
        b.metric(
            "Precio YES",
            f"{cents:.1f}¢" if cents is not None else "N/D"
        )
        c.metric(
            "Vencimiento UTC",
            expiry.strftime("%H:%M:%S") if expiry is not None else "N/D"
        )
        st.caption(f"Ticker: {contract.get('ticker', 'N/D')}")

    # =================================================
    # DECISION DE DIRECCION
    # =================================================
    st.subheader("Senal de direccion")

    minimum_score = st.selectbox(
        "Fuerza minima de indicadores",
        [2, 3, 4, 5],
        index=1
    )

    signal = "ESPERAR"
    reason = "No hay suficiente confirmacion tecnica."

    if stale:
        reason = "La cotizacion EUR/USD esta marcada como desactualizada."
    elif contract is None:
        reason = "No se ha identificado un contrato de Kalshi."
    elif target is None:
        reason = "Falta identificar el objetivo exacto del contrato."
    elif score >= minimum_score:
        signal = "BUY UP"
        reason = "Los indicadores favorecen una continuacion alcista."
    elif score <= -minimum_score:
        signal = "BUY DOWN"
        reason = "Los indicadores favorecen una continuacion bajista."

    css = {
        "BUY UP": "up",
        "BUY DOWN": "down",
        "ESPERAR": "wait"
    }[signal]

    st.markdown(
        f'<div class="signal {css}">{signal}</div>',
        unsafe_allow_html=True
    )
    st.write(reason)

    if contract is not None and target is not None:
        delta = price - target
        st.metric("Distancia al objetivo", f"{delta:+,.5f}")

        if signal == "BUY UP" and price < target:
            st.info("El precio aun esta por debajo del objetivo; la señal estima que puede superarlo.")
        elif signal == "BUY DOWN" and price > target:
            st.info("El precio aun esta por encima del objetivo; la señal estima que puede quedar por debajo.")

    # =================================================
    # ALERTA DE CAMBIO DE DIRECCION
    # =================================================
    st.subheader("Vigilancia de la senal")

    if signal == "BUY UP" and score <= 0:
        st.warning("SEÑAL EN RIESGO: los indicadores dejaron de apoyar BUY UP.")
    elif signal == "BUY DOWN" and score >= 0:
        st.warning("SEÑAL EN RIESGO: los indicadores dejaron de apoyar BUY DOWN.")
    elif signal != "ESPERAR":
        st.success("La puntuacion tecnica aun favorece la direccion actual.")
    else:
        st.info("Esperando una configuracion tecnica mas clara.")

    # =================================================
    # TAKE PROFIT INFORMATIVO
    # =================================================
    st.subheader("Aviso de TAKE PROFIT")

    entry_cents = st.number_input(
        "Tu precio de entrada en centavos",
        min_value=1.0, max_value=99.0, value=50.0, step=1.0
    )
    target_cents = st.number_input(
        "Precio del contrato para avisar",
        min_value=1.0, max_value=99.0, value=70.0, step=1.0
    )

    if cents is not None:
        st.metric("Precio YES consultado", f"{cents:.1f}¢")
        if cents >= target_cents:
            st.success(
                "TAKE PROFIT: el precio YES alcanzo el nivel configurado. "
                "Comprueba el precio de venta real y las comisiones."
            )
        elif cents <= entry_cents - 10:
            st.warning(
                "El precio YES esta al menos 10 centavos por debajo de tu entrada. "
                "Revisa el riesgo de tu posicion."
            )
        else:
            st.info("El nivel de TAKE PROFIT configurado aun no se ha alcanzado.")
    else:
        st.caption("El precio del contrato no esta disponible; no se calcula el profit.")

    # =================================================
    # GRAFICA
    # =================================================
    st.subheader("Grafica de 1 minuto")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["close"],
        name=asset, line=dict(color="white", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema9"],
        name="EMA 9", line=dict(color="#4cff9b")
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema21"],
        name="EMA 21", line=dict(color="#c69aff")
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema50"],
        name="EMA 50", line=dict(color="#ffcf70")
    ))

    if target is not None:
        fig.add_hline(
            y=target,
            line_dash="dash",
            line_color="#ffcf70",
            annotation_text="Objetivo Kalshi"
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#190f2d",
        plot_bgcolor="#190f2d",
        height=430,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="panel">Datos publicos · Analisis experimental · '
        'No ejecuta operaciones. Las señales no garantizan resultados.</div>',
        unsafe_allow_html=True
    )

except Exception as e:
    st.error("No se pudieron cargar los datos del mercado.")
    st.caption(f"Detalle tecnico: {type(e).__name__}: {e}")
