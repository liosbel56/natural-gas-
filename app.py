
import streamlit as st
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="NATGAS 15M Bot", page_icon="📈")

API = "https://api.elections.kalshi.com/trade-api/v2"

st.title("NATGAS 15 MIN")
st.caption("Monitor experimental de Kalshi")

if st.button("🔄 Buscar mercado activo", type="primary"):
    st.session_state["buscar"] = True

if st.session_state.get("buscar", False):
    try:
        response = requests.get(
            f"{API}/markets",
            params={
                "series_ticker": "KXNATGAS15M",
                "status": "open",
                "limit": 100,
            },
            timeout=15,
        )
        response.raise_for_status()
        markets = response.json().get("markets", [])

        now = datetime.now(timezone.utc)
        active = []

        for market in markets:
            close_text = market.get("close_time")
            if not close_text:
                continue

            close_time = datetime.fromisoformat(
                close_text.replace("Z", "+00:00")
            )

            if close_time > now:
                active.append((close_time, market))

        if not active:
            st.warning(
                "No se encontró una ronda abierta. "
                "Puede ser una pausa del mercado o de la API."
            )
            st.info("Vuelve a intentarlo en unos segundos.")

        else:
            close_time, market = min(active, key=lambda x: x[0])
            ticker = market.get("ticker", "No disponible")

            seconds_left = max(
                0, int((close_time - now).total_seconds())
            )
            elapsed = max(0, 900 - seconds_left)
            minute = min(15, elapsed // 60)

            st.success("Contrato activo detectado")
            st.write("**Contrato:**", ticker)
            st.write("**Minutos transcurridos:**", f"{minute}/15")
            st.write(
                "**Tiempo restante:**",
                f"{seconds_left // 60} min {seconds_left % 60} s",
            )
            st.write(
                "**Precio objetivo de apertura:**",
                market.get("floor_strike", "No publicado"),
            )

            yes_ask = market.get("yes_ask")
            no_ask = market.get("no_ask")

            st.subheader("Análisis de la ronda")

            if yes_ask is not None:
                st.metric("Comprar YES (centavos)", yes_ask)
            if no_ask is not None:
                st.metric("Comprar NO (centavos)", no_ask)

            if minute < 9:
                st.info(
                    "ANALIZANDO: espera al minuto 9. "
                    "Todavía no hay señal."
                )
            elif yes_ask is None or no_ask is None:
                st.warning(
                    "NO OPERAR: faltan precios de compra válidos."
                )
            elif yes_ask >= 60 and yes_ask < 100:
                st.success(
                    "SEÑAL EXPERIMENTAL: el mercado favorece YES."
                )
                st.caption(
                    "No confirma que el precio de Natural Gas vaya a subir."
                )
            elif no_ask >= 60 and no_ask < 100:
                st.error(
                    "SEÑAL EXPERIMENTAL: el mercado favorece NO."
                )
                st.caption(
                    "No confirma que el precio de Natural Gas vaya a bajar."
                )
            else:
                st.warning(
                    "NO OPERAR: no hay una ventaja clara con este filtro."
                )

            st.caption(
                "La señal usa precios de Kalshi, no un modelo predictivo "
                "validado. Comprueba el contrato y sus reglas antes de operar."
            )

    except requests.RequestException as error:
        st.error(f"Error conectando con Kalshi: {error}")
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Error leyendo los datos del mercado: {error}")
