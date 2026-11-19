"""Dashboard Streamlit. Ejecutar:  streamlit run app.py

Arranca el motor en un hilo de fondo (solo señales y paper trading; NUNCA envía órdenes reales).
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from btcbot.bootstrap import build_engine
from btcbot.dashboard_data import (SIGNAL_STYLE, equity_curve, last_n_table, paper_equity, recent_closes,
                                   signals_view)
from btcbot.signal_engine import WAIT

st.set_page_config(page_title="Kalshi BTC 15m — Señales", page_icon="📈", layout="wide")


@st.cache_resource
def get_engine():
    engine = build_engine("config.toml")
    engine.start_background()
    return engine


engine = get_engine()
tracker, db = engine.tracker, engine.collector.db

st.markdown(
    """<style>
    .block-container{padding-top:1.2rem}
    .badge{display:inline-block;padding:.5rem 1.2rem;border-radius:12px;color:white;font-size:1.8rem;font-weight:700}
    .small{color:#8a8f98;font-size:.85rem}
    </style>""",
    unsafe_allow_html=True,
)
st.title("Kalshi BTC 15-minute — Panel de señales")
st.caption("Modo PAPER / SHADOW: el sistema no ejecuta operaciones reales. Los resultados históricos del backtest "
           "usan un precio de mercado proxy y no prueban rentabilidad real en Kalshi.")


@st.fragment(run_every=2)
def live_panel() -> None:
    s = engine.snapshot_state()
    if s.health is None:
        st.info("Iniciando recolección de datos…")
        return
    if not s.health.ok or s.signal is None:
        st.error(f"DATA UNAVAILABLE — señales bloqueadas. {s.status_note}")
        for r in s.health.reasons:
            st.write(f"• {r}")
        return

    sig, k = s.signal, s.kalshi
    color, arrow = SIGNAL_STYLE[sig.signal]
    left = k.seconds_left(s.ts) if k else 0
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    c1.markdown(f"<div class='badge' style='background:{color}'>{arrow} {sig.signal}</div>", unsafe_allow_html=True)
    c2.metric("BTC (mediana exchanges)", f"${s.reference_price:,.2f}")
    c3.metric("Contrato", k.ticker if k else "—", f"{int(left // 60)}:{int(left % 60):02d} restantes")
    c4.metric("Strike (referencia)", f"${k.strike:,.2f}" if k and k.strike else "—")

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("P(UP) modelo", f"{sig.p_up:.1%}", f"{s.p_source}")
    p2.metric("P(DOWN) modelo", f"{sig.p_down:.1%}")
    implied = k.implied_yes if k else None
    p3.metric("Implícita Kalshi (YES)", f"{implied:.1%}" if implied is not None else "—")
    p4.metric("Edge neto esperado", f"{sig.edge_net:+.3f}" if sig.edge_net is not None else "—",
              f"requerido {sig.required_edge:.3f}" if sig.required_edge is not None else None)
    p5.metric("Confidence (0-100)", f"{sig.confidence:.0f}")

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Riesgo de reversión", f"{sig.reversal_risk:.0%}")
    r2.metric("Riesgo lateral", f"{sig.sideways_risk:.0%}")
    r3.metric("Riesgo de liquidez", f"{sig.liquidity_risk:.0%}")
    r4.metric("Régimen", s.regime)
    r5.metric("Order book imbalance", f"{s.imbalance:+.2f}" if s.imbalance is not None else "n/d")

    if sig.signal == WAIT and s.wait_streak_seconds > 300:
        st.warning(f"Llevo {s.wait_streak_seconds/60:.0f} min en WAIT: el mercado no ofrece ventaja neta tras comisiones. "
                   "No se bajan los umbrales artificialmente.")
    st.markdown("**Motivos**")
    for r in sig.reasons:
        st.write(f"• {r}")
    if s.status_note:
        st.info(s.status_note)

    f = s.features
    if f:
        with st.expander("Variables del modelo en este instante"):
            st.dataframe(pd.DataFrame([f]).T.rename(columns={0: "valor"}))

    closes = recent_closes(db, 120)
    if not closes.empty:
        chart = closes.copy()
        if k and k.strike:
            chart["strike"] = k.strike
        st.line_chart(chart, height=260)

    st.caption(f"Capital paper: ${s.bankroll:,.2f}  ·  actualizado {time.strftime('%H:%M:%S')}")


tab_live, tab_perf, tab_sig, tab_health = st.tabs(["En vivo", "Rendimiento", "Señales", "Salud de datos"])

with tab_live:
    live_panel()

with tab_perf:
    st.subheader("Rendimiento SHADOW (1 contrato por entrada, tras comisión)")
    perf = tracker.performance()
    if perf["n"] == 0:
        st.info("Aún no hay operaciones liquidadas. El resultado llega cuando Kalshi liquida cada contrato.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Operaciones", perf["n"])
        m2.metric("Win rate", f"{perf['win_rate']:.1%}", f"IC95% {perf['ci_low']:.0%}–{perf['ci_high']:.0%}")
        m3.metric("PnL neto / contrato-op.", f"{perf['net_pnl']:+.2f}")
        m4.metric("Profit factor", f"{perf['profit_factor']:.2f}")
        m5.metric("Max drawdown", f"{perf['max_drawdown']:.2f}")
        a, b = st.columns(2)
        for col, n in ((a, 50), (b, 100)):
            col.markdown(f"**Últimas {n}**")
            col.dataframe(last_n_table(tracker, n), hide_index=True)
        st.markdown("**Evolución del PnL shadow acumulado**")
        st.line_chart(equity_curve(tracker))
        st.markdown("**Evolución del capital paper (con gestión de riesgo)**")
        st.line_chart(paper_equity(engine.account.history))
        for label, col in (("Por horario (UTC)", "hour_utc"), ("Por lado", "side"), ("Por estrategia", "strategy"),
                           ("Por régimen", "regime")):
            st.markdown(f"**{label}**")
            st.dataframe(tracker.by(col), hide_index=True)

with tab_sig:
    sv = signals_view(tracker, 100)
    st.dataframe(sv, hide_index=True) if not sv.empty else st.info("Sin señales todavía.")

with tab_health:
    s = engine.snapshot_state()
    if s.health:
        st.write(f"Estado: **{s.health.status}**")
        st.write(f"Fuentes frescas: {', '.join(s.health.fresh_sources) or 'ninguna'}")
        if s.health.divergence_bps is not None:
            st.write(f"Divergencia entre exchanges: {s.health.divergence_bps:.1f} bps")
        for r in s.health.reasons:
            st.write(f"• {r}")
        if s.snapshot:
            st.dataframe(pd.DataFrame([{"fuente": q.source, "precio": q.price, "bid": q.bid, "ask": q.ask,
                                        "spread_bps": round(q.spread_bps, 2)} for q in s.snapshot.quotes.values()]),
                         hide_index=True)
