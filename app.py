
import streamlit as st
import requests
from datetime import datetime, timezone

st.set_page_config(
    page_title="Natural Gas | Kalshi Signal Bot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Natural Gas — Kalshi Signal Bot")
st.caption("Panel de análisis de 15 minutos · Solo señales, sin órdenes automáticas")

st.warning(
    "Esta versión consulta mercados públicos de Kalshi cuando están disponibles. "
    "No garantiza predicciones ni representa una conexión a precios de futuros "
    "en tiempo real. Las señales requieren datos suficientes."
)

API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

if st.button("🔄 Buscar mercados de Natural Gas"):
    st.session_state["search_markets"] = True

if st.session_state.get("search_markets", False):
    with st.spinner("Consultando mercados públicos..."):
        try:
            response = requests.get(
                API_URL,
                params={"status": "open", "limit": 200},
                timeout=15
            )
            response.raise_for_status()
            payload = response.json()
            markets = payload.get("markets", [])

            gas_markets = [
                market for market in markets
                if "natural gas" in (
                    str(market.get("title", "")) + " " +
                    str(market.get("subtitle", "")) + " " +
                    str(market.get("ticker", ""))
                ).lower()
            ]

            if gas_markets:
                st.success(f"Mercados encontrados: {len(gas_markets)}")
                for market in gas_markets:
                    with st.container(border=True):
                        st.subheader(
                            market.get("title")
                            or market.get("subtitle")
                            or market.get("ticker", "Mercado")
                        )
                        st.write("Ticker:", market.get("ticker", "No disponible"))
                        st.write(
                            "Precio YES:",
                            market.get("yes_ask", "No disponible")
                        )
                        st.write(
                            "Precio NO:",
                            market.get("no_ask", "No disponible")
                        )
                        st.write(
                            "Cierre:",
                            market.get("close_time", "No disponible")
                        )
            else:
                st.info(
                    "No se encontraron mercados abiertos con el nombre "
                    "'Natural Gas' en los primeros resultados. Esto no "
                    "significa que no exista uno. Busca también en Kalshi."
                )

        except requests.RequestException as error:
            st.error(f"No se pudo consultar la API pública: {error}")
        except (ValueError, TypeError):
            st.error("La respuesta recibida no tenía el formato esperado.")

st.divider()
st.subheader("⏱️ Ciclo de análisis de 15 minutos")

now = datetime.now(timezone.utc)
minute = now.minute % 15
remaining = 15 - minute

col1, col2, col3 = st.columns(3)
col1.metric("Minuto del ciclo", f"{minute}/15")
col2.metric("Minutos restantes", remaining)
col3.metric("Hora UTC", now.strftime("%H:%M:%S"))

if minute < 9:
    st.info("FASE: recopilación y análisis. Todavía no hay señal automática.")
elif minute < 15:
    st.info(
        "FASE: seguimiento. No se genera BUY UP o BUY DOWN sin datos "
        "de precio e historial suficientes."
    )

st.caption(
    "Este panel es una primera versión. No calcula todavía una probabilidad "
    "predictiva validada ni ejecuta operaciones."
)
