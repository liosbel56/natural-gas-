from __future__ import annotations
import time
import streamlit as st
import pandas as pd
from btcbot.engine import BotEngine
from btcbot.dashboard_data import signal_history_frame, equity_frame

st.set_page_config(page_title="Kalshi BTC 15m Signals", page_icon="📈", layout="wide")
st.title("Kalshi BTC 15-minute — Señales")
st.caption("PAPER / SHADOW solamente. No envía órdenes. Ningún win rate está garantizado.")

@st.cache_resource
def get_engine():
    e = BotEngine("config.toml")
    e.start()
    return e

engine = get_engine()
if st.button("Actualizar ahora"):
    engine.step()

s = engine.snapshot()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Señal", s["signal"])
c2.metric("BTC", f"${s['btc']:,.2f}" if s["btc"] else "—")
c3.metric("Confianza", f"{s['confidence']:.0f}/100" if s["confidence"] is not None else "—")
c4.metric("Contrato Kalshi", s["ticker"] or "No detectado")

st.subheader("Probabilidades y ventaja")
a,b,c,d = st.columns(4)
a.metric("P(UP)", f"{s['p_up']:.1%}" if s["p_up"] is not None else "—")
b.metric("P(DOWN)", f"{s['p_down']:.1%}" if s["p_down"] is not None else "—")
c.metric("Edge neto", f"{s['edge']:+.3f}" if s["edge"] is not None else "—")
d.metric("Régimen", s["regime"])
if s["status"]:
    st.info(s["status"])
if s["reasons"]:
    st.markdown("**Motivos**")
    for reason in s["reasons"]:
        st.write("• " + reason)

left,right = st.columns(2)
with left:
    st.subheader("BTC — velas recientes")
    if not engine.candles.empty:
        st.line_chart(engine.candles.set_index("time")["close"])
with right:
    st.subheader("Indicadores")
    if s["features"]:
        st.dataframe(pd.DataFrame([s["features"]]).T.rename(columns={0:"Valor"}))

st.subheader("Historial de señales")
hist = signal_history_frame(engine.history)
if hist.empty:
    st.caption("Aún no hay señales registradas.")
else:
    st.dataframe(hist.tail(100), use_container_width=True, hide_index=True)

st.subheader("Rendimiento paper")
m1,m2,m3 = st.columns(3)
perf = engine.performance()
m1.metric("Señales registradas", perf["n"])
m2.metric("Win rate observado", f'{perf["win_rate"]:.1%}' if perf["win_rate"] is not None else "—")
m3.metric("PnL paper", f'${perf["pnl"]:+.2f}')
eq = equity_frame(engine.history)
if not eq.empty:
    st.line_chart(eq.set_index("trade")["equity"])
st.caption("Actualizado: " + time.strftime("%H:%M:%S") + " · Recarga la página para refrescar.")
