# -*- coding: utf-8 -*-
"""
Analizador de señales — Kalshi BTC 15 minutos (serie KXBTC15M)  ·  versión móvil + diagnóstico
===============================================================================================
* Es un ANALIZADOR. No compra ni vende nada. No necesita claves de API.
* Datos de precio: Coinbase Exchange (API pública).
* Datos del contrato: Kalshi (API pública de mercado, sin autenticación).
* Los resultados pasados NO garantizan resultados futuros. Esto no es asesoría financiera.

Novedades de esta versión
-------------------------
* Panel de diagnóstico: estado de CADA filtro (valor leído, límite, motivo), probabilidades leídas,
  indicadores, horas de actualización y valores crudos de Kalshi.
* Se evalúan todos los filtros a la vez (antes el primero que fallaba ocultaba a los demás).
* Interfaz pensada para celular (una columna, tarjetas, tablas compactas, gráfica estática).
* Correcciones: ver la lista "CAMBIOS" en la ayuda de la propia app.

Ejecutar:   streamlit run app.py
"""
from __future__ import annotations

import html as _html
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
COINBASE_BASE = "https://api.exchange.coinbase.com"
KALSHI_BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
PRODUCT = "BTC-USD"
SERIES = os.getenv("KALSHI_SERIES", "KXBTC15M")
WINDOW_MIN = 15
WARMUP = 100                  # velas cerradas mínimas para indicadores fiables
MIN_SIGMA = 1.5e-4            # piso de volatilidad por minuto (log-retorno)
HISTORY_FILE = "signal_history.csv"
DEFAULT_TZ = os.getenv("APP_TZ", "America/Chicago")
LOCAL_TZ = ZoneInfo(DEFAULT_TZ)
TZ_CHOICES = ["America/Chicago", "America/New_York", "America/Denver", "America/Los_Angeles",
              "America/Mexico_City", "America/Bogota", "America/Lima", "America/Santiago",
              "America/Argentina/Buenos_Aires", "America/Caracas", "Europe/Madrid", "UTC"]
UA = {"User-Agent": "kalshi-btc15-analyzer/1.1", "Accept": "application/json"}
PRICE_KEYS = ("yes_bid", "yes_ask", "no_bid", "no_ask")
CANDLE_MAX_LAG_S = 150        # retraso máximo aceptado de la última vela cerrada

GROUPS = [  # (etiqueta, columna, peso)  — 5 grupos, no 12 indicadores sueltos
    ("Tendencia (EMA 9/21/50 y pendiente de EMA 21)", "g_trend", 0.25),
    ("Momentum (histograma MACD y momentum 3/5 velas)", "g_mom", 0.30),
    ("Sobreextensión RSI(14) (contraria)", "g_rsi", 0.10),
    ("Volumen que confirma el movimiento", "g_vol", 0.15),
    ("Estructura (rupturas, soportes/resistencias)", "g_struct", 0.20),
]

# Filtros de la fase de decisión (se usan también para rellenar "no evaluado" cuando faltan datos)
DECISION_FILTERS = [
    ("quotes", "Precios de compra de Kalshi"),
    ("time_left", "Tiempo restante"),
    ("edge", "Ventaja neta mínima"),
    ("entry", "Rango del precio de entrada"),
    ("spread", "Spread máximo"),
    ("uncertainty", "Incertidumbre"),
]


@dataclass
class Cfg:
    refresh_s: int = 8
    min_edge: float = 0.04        # ventaja mínima neta (en $ por contrato)
    uncal_extra: float = 0.02     # margen extra mientras el modelo no esté validado
    max_spread: float = 0.08
    min_entry: float = 0.05
    max_entry: float = 0.95
    fee_rate: float = 0.07        # comisión taker general de Kalshi: rate*P*(1-P)
    slippage: float = 0.01
    min_tau_min: float = 0.75     # no emitir señal con menos de 45 s
    basis_bps: float = 1.5        # incertidumbre Coinbase vs BRTI (puntos base)
    drift_cap: float = 0.25       # máximo desplazamiento (en desv. típicas) que puede aportar el score
    sigma_inflate: float = 1.10   # colas gruesas
    max_age_s: int = 20           # antigüedad máxima aceptada de precio/Kalshi (tiempo desde la descarga)
    max_trade_age_s: int = 60     # NUEVO: antigüedad máxima de la última operación de Coinbase
    max_clock_skew_s: float = 8.0  # desfase máximo del reloj local frente a los servidores
    max_ref_dev: float = 0.03     # NUEVO: |ln(precio/referencia)| máximo creíble en 15 min (3 %)
    quote_tol: float = 0.02       # NUEVO: tolerancia de coherencia entre precios YES y NO ($)
    chart_minutes: int = 60
    use_backtest: bool = True
    sim_half_spread: float = 0.01


# ──────────────────────────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.4, status_forcelist=(500, 502, 503, 504),
                  allowed_methods=("GET",), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def http_get(base: str, path: str, params: Optional[dict] = None, timeout: int = 8
             ) -> Tuple[Any, float, Optional[datetime]]:
    t0 = time.time()
    r = get_session().get(base + path, params=params, headers=UA, timeout=timeout)
    lat = (time.time() - t0) * 1000
    if r.status_code == 429:
        raise RuntimeError("Límite de solicitudes alcanzado (HTTP 429). Aumenta el intervalo de actualización.")
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError("Respuesta inválida (no es JSON).")
    server_dt = None
    try:
        server_dt = parsedate_to_datetime(r.headers["Date"])
    except Exception:
        pass
    return j, lat, server_dt


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# COINBASE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3, show_spinner=False)
def fetch_ticker() -> dict:
    j, lat, sdt = http_get(COINBASE_BASE, f"/products/{PRODUCT}/ticker")
    price = _to_float(j.get("price"))
    if price is None or price <= 0:
        raise RuntimeError("Coinbase devolvió un precio inválido.")
    ex_time = None
    try:
        ex_time = pd.to_datetime(j.get("time"), utc=True).to_pydatetime()
    except Exception:
        pass
    return {"price": price, "bid": _to_float(j.get("bid")), "ask": _to_float(j.get("ask")),
            "exchange_time": ex_time, "fetched_at": utcnow(), "latency_ms": lat, "server_date": sdt}


def parse_candles(raw: Any) -> pd.DataFrame:
    """Coinbase: [[time, low, high, open, close, volume], ...] (más reciente primero)."""
    if not isinstance(raw, list) or len(raw) == 0:
        raise RuntimeError("Coinbase devolvió velas vacías o en formato inesperado.")
    try:
        arr = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        raise RuntimeError("Velas con valores no numéricos.")
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise RuntimeError("Las velas no tienen las 6 columnas esperadas.")
    df = pd.DataFrame(arr[:, :6], columns=["ts", "low", "high", "open", "close", "volume"])
    df = df[np.isfinite(df).all(axis=1)]
    df = df[(df[["low", "high", "open", "close"]] > 0).all(axis=1)]
    df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9)
            & (df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9)
            & (df["volume"] >= 0)]
    if df.empty:
        raise RuntimeError("Todas las velas recibidas fueron inválidas.")
    df["time"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    df = df.drop(columns="ts").drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


def fill_gaps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Rellena minutos sin operaciones (Coinbase los omite): vela plana, volumen 0."""
    d = df.set_index("time")
    full = pd.date_range(d.index[0], d.index[-1], freq="60s")
    missing = len(full) - len(d)
    if missing == 0:
        return df.reset_index(drop=True), 0
    d = d.reindex(full)
    d["close"] = d["close"].ffill()
    for col in ("open", "high", "low"):
        d[col] = d[col].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0.0)
    d.index.name = "time"
    return d.reset_index(), missing


@st.cache_data(ttl=5, show_spinner=False)
def fetch_candles() -> dict:
    j, lat, sdt = http_get(COINBASE_BASE, f"/products/{PRODUCT}/candles", params={"granularity": 60})
    df = parse_candles(j)
    return {"df": df, "fetched_at": utcnow(), "latency_ms": lat, "server_date": sdt}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(hours: int) -> pd.DataFrame:
    """Historial de velas de 1 min paginando (máx. ~290 velas por solicitud)."""
    end = utcnow().replace(second=0, microsecond=0)
    remaining = int(hours) * 60
    frames = []
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    while remaining > 0:
        n = min(290, remaining)
        start = end - timedelta(minutes=n)
        j, _, _ = http_get(COINBASE_BASE, f"/products/{PRODUCT}/candles",
                           params={"granularity": 60, "start": start.strftime(fmt), "end": end.strftime(fmt)},
                           timeout=15)
        if isinstance(j, list) and j:
            frames.append(parse_candles(j))
        end = start
        remaining -= n
        time.sleep(0.2)  # cortesía con el límite de la API pública
    if not frames:
        raise RuntimeError("No se pudo descargar historial de Coinbase.")
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    # descartar la vela que aún se está formando
    df = df[df["time"] + pd.Timedelta(seconds=60) <= pd.Timestamp(utcnow())].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# KALSHI
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=4, show_spinner=False)
def fetch_kalshi_markets() -> dict:
    j, lat, sdt = http_get(KALSHI_BASE, "/markets",
                           params={"series_ticker": SERIES, "status": "open", "limit": 50})
    markets = j.get("markets") if isinstance(j, dict) else None
    if not isinstance(markets, list):
        raise RuntimeError("Kalshi no devolvió la lista 'markets'.")
    return {"markets": [parse_market(m) for m in markets], "fetched_at": utcnow(),
            "latency_ms": lat, "server_date": sdt}


def fetch_kalshi_market_result(ticker: str) -> Optional[str]:
    j, _, _ = http_get(KALSHI_BASE, f"/markets/{ticker}")
    m = j.get("market", j) if isinstance(j, dict) else {}
    res = str(m.get("result") or "").lower()
    return res if res in ("yes", "no") else None


def _kprice(m: dict, key: str) -> Optional[float]:
    """Kalshi expone precios en dólares (*_dollars, texto). Desde marzo de 2026 los campos en centavos
    enteros ya no existen; se mantiene el respaldo por si algún entorno los devuelve.
    Un precio 0 o 1 significa "sin órdenes" y se descarta (se deriva del lado contrario)."""
    v = _to_float(m.get(f"{key}_dollars"))
    if v is None:
        c = _to_float(m.get(key))
        v = None if c is None else c / 100.0
    if v is None or not (0.0 < v < 1.0):
        return None
    return v


def _knum(m: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = _to_float(m.get(k))
        if v is not None:
            return v
    return None


def _ktime(x: Any) -> Optional[datetime]:
    if not x:
        return None
    try:
        return pd.to_datetime(x, utc=True).to_pydatetime()
    except Exception:
        return None


def parse_market(m: dict) -> dict:
    q = {k: _kprice(m, k) for k in PRICE_KEYS}
    direct = [k for k in PRICE_KEYS if q[k] is not None]
    # En un mercado binario: ask(YES) = 1 - bid(NO) y viceversa.
    if q["yes_ask"] is None and q["no_bid"] is not None:
        q["yes_ask"] = round(1 - q["no_bid"], 4)
    if q["no_ask"] is None and q["yes_bid"] is not None:
        q["no_ask"] = round(1 - q["yes_bid"], 4)
    if q["yes_bid"] is None and q["no_ask"] is not None:
        q["yes_bid"] = round(1 - q["no_ask"], 4)
    if q["no_bid"] is None and q["yes_ask"] is not None:
        q["no_bid"] = round(1 - q["yes_ask"], 4)
    derived = [k for k in PRICE_KEYS if k not in direct and q[k] is not None]
    rules = ((m.get("rules_primary") or "") + " " + (m.get("rules_secondary") or "")).strip()
    raw_keys = [f"{k}{suf}" for k in PRICE_KEYS for suf in ("_dollars", "")] + [
        "last_price_dollars", "yes_ask_size_fp", "yes_bid_size_fp", "status", "strike_type",
        "floor_strike", "open_time", "close_time", "updated_time", "price_level_structure"]
    raw = {k: m.get(k) for k in raw_keys if k in m}
    liq = _knum(m, ["liquidity_dollars"])
    if liq is None:
        liq_c = _knum(m, ["liquidity"])
        liq = None if liq_c is None else liq_c / 100.0
    return {
        "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "title": m.get("title") or "",
        "subtitle": m.get("yes_sub_title") or "", "status": m.get("status"),
        "open_time": _ktime(m.get("open_time")), "close_time": _ktime(m.get("close_time")),
        "expiration_time": _ktime(m.get("expiration_time") or m.get("latest_expiration_time")),
        "updated": _ktime(m.get("updated_time")),
        "strike": _to_float(m.get("floor_strike")), "strike_type": (m.get("strike_type") or "").lower() or None,
        "rules": rules, "yes_bid": q["yes_bid"], "yes_ask": q["yes_ask"], "no_bid": q["no_bid"], "no_ask": q["no_ask"],
        "last_price": _kprice(m, "last_price"),
        "yes_ask_size": _knum(m, ["yes_ask_size_fp", "yes_ask_size"]),
        "yes_bid_size": _knum(m, ["yes_bid_size_fp", "yes_bid_size"]),
        "volume": _knum(m, ["volume_fp", "volume"]), "open_interest": _knum(m, ["open_interest_fp", "open_interest"]),
        "liquidity": liq, "derived": derived, "raw": raw,
    }


def quote_check(q: dict, tol: float) -> Tuple[str, str]:
    """Coherencia de los 4 precios. En Kalshi: ask(YES)+bid(NO) = 1 y bid(YES)+ask(NO) = 1."""
    ya, yb, na, nb = q.get("yes_ask"), q.get("yes_bid"), q.get("no_ask"), q.get("no_bid")
    issues: List[str] = []
    if ya is not None and yb is not None and yb > ya + 1e-9:
        issues.append(f"YES cruzado (venta {cents(yb)} > compra {cents(ya)})")
    if na is not None and nb is not None and nb > na + 1e-9:
        issues.append(f"NO cruzado (venta {cents(nb)} > compra {cents(na)})")
    if ya is not None and nb is not None and abs(ya + nb - 1) > tol:
        issues.append(f"YES compra {cents(ya)} + NO venta {cents(nb)} ≠ 100¢")
    if yb is not None and na is not None and abs(yb + na - 1) > tol:
        issues.append(f"YES venta {cents(yb)} + NO compra {cents(na)} ≠ 100¢")
    return ("fail" if issues else "ok"), "; ".join(issues)


@dataclass
class Contract:
    ticker: str
    title: str
    close_time: datetime
    window_start: datetime
    reference: Optional[float]
    reference_kind: str
    yes_if_above: bool
    supported: bool
    notes: List[str]
    rules: str


def build_contract(m: dict, closed: pd.DataFrame, now: datetime) -> Contract:
    """Interpreta las reglas del contrato. No asume que todos los contratos son iguales."""
    close = m["close_time"]
    wstart = close - timedelta(minutes=WINDOW_MIN)
    text = (m["rules"] + " " + m["title"]).lower()
    notes: List[str] = []
    supported = True
    ref: Optional[float] = None
    kind = "desconocido"
    above_words = re.search(r"\b(at least|at or above|above|greater|higher|more than|up)\b", text)
    below_words = re.search(r"\b(below|less than|lower|under|down)\b", text)

    if "brti" in text or "cf benchmarks" in text:
        notes.append("Liquida con CF Benchmarks BRTI (promedio de ~60 s). Coinbase es solo una aproximación.")
    elif text.strip():
        notes.append("Las reglas no mencionan BRTI: verifica manualmente la fuente de liquidación.")

    if m["strike"] is not None:
        ref, kind = m["strike"], "umbral fijo (floor_strike)"
        st_type = m["strike_type"]
        if st_type in ("greater", "greater_or_equal"):
            yes_above = True
        elif st_type in ("less", "less_or_equal"):
            yes_above = False
        elif st_type is None and above_words and not below_words:
            yes_above = True
        elif st_type is None and below_words and not above_words:
            yes_above = False
        else:
            yes_above, supported = True, False
            notes.append(f"Tipo de umbral no soportado o ambiguo ({st_type!r}); no se emite señal.")
    else:
        kind = "promedio 60 s previos al inicio de la ventana"
        yes_above = True
        if not (above_words and "average" in text):
            supported = False
            notes.append("No se reconoce la regla (¿'promedio al cierre ≥ promedio al inicio'?); no se emite señal.")
        if now < wstart:
            notes.append("La ventana aún no empieza: la referencia todavía no existe.")
        else:
            row = closed[closed["time"] == pd.Timestamp(wstart - timedelta(seconds=60))]
            if len(row):
                r = row.iloc[0]
                ref = float((r["open"] + r["close"]) / 2.0)   # aproximación del promedio de 60 s
                notes.append("Referencia aproximada con la vela de 1 min previa al inicio (no es el BRTI exacto).")
            else:
                notes.append("No hay vela de Coinbase para el minuto previo al inicio de la ventana.")
    return Contract(m["ticker"], m["title"], close, wstart, ref, kind, yes_above, supported, notes, m["rules"])


def select_market(markets: List[dict], now: datetime, forced: Optional[str]) -> Optional[dict]:
    cands = sorted([m for m in markets if m.get("close_time") and m["close_time"] > now],
                   key=lambda m: m["close_time"])
    if forced:
        for m in cands:
            if m["ticker"] == forced:
                return m
    running = [m for m in cands if m["close_time"] - timedelta(minutes=WINDOW_MIN) <= now]
    if running:
        return running[0]
    return cands[0] if cands else None


# ──────────────────────────────────────────────────────────────────────────────
# INDICADORES (todos causales: la fila t sólo usa datos hasta t)
# ──────────────────────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    for n in (9, 21, 50):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    rsi = rsi.where(~((loss == 0) & (gain == 0)), 50.0)
    d["rsi"] = rsi

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    ret = np.log(c / c.shift(1))
    d["ret"] = ret
    d["mom3"] = np.log(c / c.shift(3))
    d["mom5"] = np.log(c / c.shift(5))

    ewma_var = (ret ** 2).ewm(alpha=0.06, adjust=False).mean()
    roll_var = (ret ** 2).rolling(60, min_periods=30).mean()
    d["sigma"] = np.sqrt(0.5 * ewma_var + 0.5 * roll_var).clip(lower=MIN_SIGMA)
    d["sigma_ratio"] = np.sqrt(ewma_var / roll_var.replace(0, np.nan))

    vol_ma = v.rolling(20, min_periods=10).mean()
    d["vol_ratio"] = v / vol_ma.replace(0, np.nan)
    d["vol_chg"] = v.pct_change().replace([np.inf, -np.inf], np.nan)

    d["high20"] = h.rolling(20).max().shift(1)
    d["low20"] = l.rolling(20).min().shift(1)
    d["resist"] = h.rolling(30).max().shift(1)
    d["support"] = l.rolling(30).min().shift(1)
    d["break_up"] = (c > d["high20"]).astype(float)
    d["break_dn"] = (c < d["low20"]).astype(float)

    atr_s = d["atr"].replace(0, np.nan)
    d["dist_res_atr"] = (d["resist"] - c) / atr_s
    d["dist_sup_atr"] = (c - d["support"]) / atr_s

    # ── 5 grupos de evidencia, cada uno en [-1, 1] ──
    align = (np.sign(d["ema9"] - d["ema21"]) + np.sign(d["ema21"] - d["ema50"])) / 2
    slope = np.tanh((d["ema21"] - d["ema21"].shift(5)) / atr_s)
    d["g_trend"] = 0.5 * align + 0.5 * slope

    mh = np.tanh(3 * d["macd_hist"] / atr_s)
    m3 = np.tanh(d["mom3"] / (d["sigma"] * math.sqrt(3)) / 2)
    m5 = np.tanh(d["mom5"] / (d["sigma"] * math.sqrt(5)) / 2)
    d["g_mom"] = (mh + m3 + m5) / 3

    r = d["rsi"]
    d["g_rsi"] = pd.Series(np.where(r > 70, -(r - 70) / 30, np.where(r < 30, (30 - r) / 30, 0.0)),
                           index=d.index).clip(-1, 1)

    sign3 = np.sign(ret.rolling(3).sum())
    d["g_vol"] = sign3 * np.tanh((d["vol_ratio"] - 1).clip(lower=0))

    room = np.tanh((d["dist_res_atr"] - d["dist_sup_atr"]) / 4)
    d["g_struct"] = 0.6 * (d["break_up"] - d["break_dn"]) + 0.4 * room

    score = sum(w * d[col].fillna(0.0) for _, col, w in GROUPS)
    d["score"] = score.clip(-1, 1)

    d["valid"] = ((np.arange(len(d)) >= WARMUP) & d["sigma"].notna() & d["atr"].notna()
                  & d["rsi"].notna() & np.isfinite(d["score"]) & (d["atr"] > 0))
    return d


# ──────────────────────────────────────────────────────────────────────────────
# MODELO
# ──────────────────────────────────────────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def model_prob(S0: float, K: float, tau_min: float, sigma_min: float, score: float, cfg: Cfg) -> dict:
    """P(promedio de los últimos 60 s del precio al vencimiento >= K).

    Movimiento browniano en log-precio. El promedio de los últimos 60 s tiene menos varianza que
    el precio puntual: t_eff = tau - 2/3 min (con piso). A la desviación se suma, en cuadratura,
    el error de base Coinbase↔BRTI. El 'score' sólo desplaza z, con tope drift_cap desviaciones.
    """
    t_eff = max(tau_min - 2.0 / 3.0, 0.2)
    var = (sigma_min * cfg.sigma_inflate) ** 2 * t_eff + (cfg.basis_bps / 1e4) ** 2
    sd = math.sqrt(var)
    dist = math.log(S0 / K)
    drift_z = cfg.drift_cap * float(score) * min(1.0, math.sqrt(t_eff / WINDOW_MIN))
    z0 = dist / sd
    z = z0 + drift_z
    clip = lambda p: min(max(p, 0.005), 0.995)
    return {"p": clip(norm_cdf(z)), "p_nodrift": clip(norm_cdf(z0)), "z": z, "dist": dist, "sd": sd,
            "t_eff": t_eff, "drift_z": drift_z}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def apply_platt(p: float, a: float, b: float) -> float:
    z = a * float(logit(np.array([p]))[0]) + b
    return float(min(max(1 / (1 + math.exp(-z)), 0.005), 0.995))


def fit_platt(p: np.ndarray, y: np.ndarray, lam: float = 1.0) -> Tuple[float, float]:
    """Regresión logística de 2 parámetros (IRLS) con regularización hacia la identidad (a=1, b=0)."""
    x = logit(p)
    X = np.column_stack([x, np.ones_like(x)])
    prior = np.array([1.0, 0.0])
    w = prior.copy()
    for _ in range(50):
        z = X @ w
        q = 1 / (1 + np.exp(-z))
        g = X.T @ (q - y) + lam * (w - prior)
        H = X.T @ (X * (q * (1 - q))[:, None]) + lam * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.linalg.norm(step) < 1e-8:
            break
    return float(w[0]), float(w[1])


# ──────────────────────────────────────────────────────────────────────────────
# FORMATO
# ──────────────────────────────────────────────────────────────────────────────
def cents(x: Optional[float]) -> str:
    """Kalshi cotiza con décimas de centavo (p. ej. 91.7¢ por encima de 90¢): no redondear a ciegas."""
    if x is None:
        return "—"
    c = x * 100
    return f"{c:.0f}¢" if abs(c - round(c)) < 0.05 else f"{c:.1f}¢"


def pct(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def fnum(x: Any, nd: int = 2, prefix: str = "", suffix: str = "") -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{prefix}{float(x):,.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def esc(x: Any) -> str:
    return _html.escape(str(x))


def mkf(fid: str, name: str, status: str, value: str = "", limit: str = "", detail: str = "",
        blocking: bool = True) -> dict:
    """Un filtro/comprobación. status: ok | fail | warn | na.  Solo 'fail' + blocking=True impide la señal."""
    return {"id": fid, "name": name, "status": status, "value": value, "limit": limit,
            "detail": detail, "blocking": blocking}


# ──────────────────────────────────────────────────────────────────────────────
# COSTES, INCERTIDUMBRE Y DECISIÓN
# ──────────────────────────────────────────────────────────────────────────────
def kalshi_fee(price: float, rate: float) -> float:
    """Comisión taker por contrato (sin redondeo): rate · P · (1 − P)."""
    return rate * price * (1 - price)


def kalshi_fee_order(contracts: int, price: float, rate: float) -> float:
    """Comisión de una orden: redondeada hacia arriba al siguiente centavo."""
    return math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100


def uncertainty(z: float, sd: float, sigma_ratio: float, spread: Optional[float],
                tau_min: float, calibrated: bool, cfg: Cfg) -> Tuple[str, List[str], int]:
    """Cada factor suma 1 punto. 0-1 → Baja · 2-3 → Media · 4 o más → Alta (bloquea la señal)."""
    pts, why = 0, []
    if not calibrated:
        pts += 1; why.append("probabilidad sin validar con backtest (+1)")
    if abs(z) < 0.5:
        pts += 1; why.append("el precio está muy cerca del umbral (+1)")
    if sigma_ratio is not None and np.isfinite(sigma_ratio) and sigma_ratio > 1.6:
        pts += 1; why.append("volatilidad reciente inestable (+1)")
    if spread is not None and spread > 0.05:
        pts += 1; why.append(f"spread amplio ({spread * 100:.0f}¢) (+1)")
    if tau_min < 2:
        pts += 1; why.append("quedan menos de 2 min: el error de referencia pesa más (+1)")
    if sd > 0 and (cfg.basis_bps / 1e4) / sd > 0.6:
        pts += 1; why.append("el error Coinbase↔BRTI es comparable al movimiento esperado (+1)")
    label = "Baja" if pts <= 1 else "Media" if pts <= 3 else "Alta"
    return label, why, pts


def decide(p_yes: float, q: dict, tau_min: float, cfg: Cfg, calibrated: bool,
           problems: List[str], unc_label: str, unc_detail: str = "") -> dict:
    """Decisión final. Evalúa TODOS los filtros (no se detiene en el primero que falla).
    Solo hay señal si no hay problemas de datos y ningún filtro bloqueante falla."""
    req = cfg.min_edge + (0.0 if calibrated else cfg.uncal_extra)
    out: Dict[str, Any] = {"signal": "ESPERAR", "side": None, "best": None, "reasons": [], "sides": {},
                           "req_edge": req, "filters": []}
    for side, p, ask, bid in (("UP", p_yes, q.get("yes_ask"), q.get("yes_bid")),
                              ("DOWN", 1 - p_yes, q.get("no_ask"), q.get("no_bid"))):
        if ask is None:
            out["sides"][side] = None
            continue
        fee = kalshi_fee(ask, cfg.fee_rate)
        be = ask + fee + cfg.slippage
        out["sides"][side] = {
            "p": p, "ask": ask, "bid": bid, "fee": fee,
            "spread": (ask - bid) if bid is not None else None,
            "breakeven": be, "edge": p - be, "req_p": be + req,
            "win": 1 - be, "loss": -be,
        }
    fl: List[dict] = out["filters"]
    valid = {k: v for k, v in out["sides"].items() if v}

    # 1) precios de compra
    if valid:
        fl.append(mkf("quotes", "Precios de compra de Kalshi", "ok",
                      " · ".join(f"{k} {cents(v['ask'])}" for k, v in valid.items())))
    else:
        fl.append(mkf("quotes", "Precios de compra de Kalshi", "fail", "sin precios", "",
                      "Kalshi no muestra precios de compra válidos."))
    # 2) tiempo restante
    t_ok = tau_min >= cfg.min_tau_min
    fl.append(mkf("time_left", "Tiempo restante", "ok" if t_ok else "fail", f"{tau_min * 60:.0f} s",
                  f"≥ {cfg.min_tau_min * 60:.0f} s",
                  "" if t_ok else f"Quedan {tau_min * 60:.0f} s: demasiado cerca del vencimiento para emitir señal."))
    side = None
    if valid:
        side, s = max(valid.items(), key=lambda kv: kv[1]["edge"])
        out["best"] = side
        # 3) ventaja neta
        e_ok = s["edge"] >= req
        imposible = s["req_p"] > 0.995
        det = (f"Mejor opción ({side}): el modelo da {pct(s['p'])} y el costo de equilibrio es {pct(s['breakeven'])} "
               f"(compra {cents(s['ask'])} + comisión {s['fee'] * 100:.2f}¢ + deslizamiento {cents(cfg.slippage)}). "
               f"Ventaja neta {s['edge'] * 100:+.1f}¢; se exigen ≥ {req * 100:.1f}¢, o sea una probabilidad del modelo "
               f"≥ {pct(s['req_p'])}"
               + (" (el modelo nunca pasa de 99.5 %: a este precio no puede haber señal)." if imposible else "."))
        fl.append(mkf("edge", f"Ventaja neta mínima ({side})", "ok" if e_ok else "fail",
                      f"{s['edge'] * 100:+.1f}¢", f"≥ {req * 100:.1f}¢", "" if e_ok else det))
        # 4) rango de precio de entrada
        r_ok = cfg.min_entry <= s["ask"] <= cfg.max_entry
        fl.append(mkf("entry", "Rango del precio de entrada", "ok" if r_ok else "fail", cents(s["ask"]),
                      f"{cfg.min_entry * 100:.0f}¢–{cfg.max_entry * 100:.0f}¢",
                      "" if r_ok else f"Precio de entrada {cents(s['ask'])} fuera del rango permitido."))
        # 5) spread
        if s["spread"] is None:
            fl.append(mkf("spread", "Spread máximo", "warn", "sin precio de venta", f"≤ {cfg.max_spread * 100:.0f}¢",
                          "No hay precio de venta para medir el spread (no bloquea).", blocking=False))
        else:
            sp_ok = s["spread"] <= cfg.max_spread
            fl.append(mkf("spread", "Spread máximo", "ok" if sp_ok else "fail", cents(s["spread"]),
                          f"≤ {cfg.max_spread * 100:.0f}¢",
                          "" if sp_ok else f"Spread de {cents(s['spread'])} supera el máximo de {cfg.max_spread * 100:.0f}¢."))
    else:
        for fid, name in DECISION_FILTERS[2:5]:
            fl.append(mkf(fid, name, "na", blocking=False))
    # 6) incertidumbre
    st_u = {"Baja": "ok", "Media": "warn", "Alta": "fail"}[unc_label]
    fl.append(mkf("uncertainty", "Incertidumbre", st_u, unc_label, "no ser 'Alta' (4+ puntos)",
                  ("Incertidumbre alta: " + unc_detail) if st_u == "fail" else unc_detail))

    fails = [f for f in fl if f["status"] == "fail" and f["blocking"]]
    if problems or fails or not valid:
        out["reasons"] = list(problems) + [f["detail"] for f in fails if f["detail"]]
        if not out["reasons"]:
            out["reasons"] = ["Sin datos suficientes."]
        return out
    s = out["sides"][side]
    out["signal"] = f"BUY {side}"
    out["side"] = side
    out["reasons"] = [f"Ventaja neta estimada {s['edge'] * 100:+.1f}¢ por contrato tras comisión y deslizamiento "
                      f"(umbral {req * 100:.1f}¢)."]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTING
# ──────────────────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def run_backtest(df_raw: pd.DataFrame, cfg: Cfg, elapsed_points=(3, 6, 9, 12), train_frac: float = 0.6) -> dict:
    """Simula ventanas de 15 min alineadas a :00/:15/:30/:45 con velas de Coinbase.

    * Sin información futura: en cada decisión sólo se usan velas cerradas hasta ese instante.
    * Train = primeros 60 % de ventanas (cronológico). Test = el resto.
    * No hay historial público de cotizaciones de Kalshi: el precio de mercado se SIMULA como la
      probabilidad del modelo sin tendencia (benchmark) ± spread + deslizamiento + comisión.
      El P&L es por tanto SINTÉTICO y sólo mide si el score aporta algo sobre ese benchmark.
    """
    df, n_gaps = fill_gaps(df_raw)
    f = compute_features(df)
    times = list(f["time"])
    pos = {t: i for i, t in enumerate(times)}
    o, c = f["open"].to_numpy(), f["close"].to_numpy()
    sigma, score, valid = f["sigma"].to_numpy(), f["score"].to_numpy(), f["valid"].to_numpy()
    sratio = f["sigma_ratio"].to_numpy()

    rows = []
    for s in times:
        if s.minute % WINDOW_MIN != 0:
            continue
        i0 = pos[s]
        if i0 < 1 or i0 + WINDOW_MIN - 1 >= len(f):
            continue
        K = (o[i0 - 1] + c[i0 - 1]) / 2.0                 # promedio ~60 s previo al inicio
        settle = (o[i0 + WINDOW_MIN - 1] + c[i0 + WINDOW_MIN - 1]) / 2.0   # ~60 s previos al cierre
        y = 1.0 if settle >= K else 0.0
        for m in elapsed_points:
            j = i0 + m - 1                                  # última vela CERRADA en la decisión
            if not valid[j]:
                continue
            mp = model_prob(c[j], K, WINDOW_MIN - m, sigma[j], score[j], cfg)
            rows.append({"window": s, "elapsed": m, "tau": WINDOW_MIN - m, "p_raw": mp["p"],
                         "p_nd": mp["p_nodrift"], "z": mp["z"], "sd": mp["sd"], "y": y,
                         "persist": 1.0 if c[j] >= K else 0.0, "sratio": sratio[j]})
    R = pd.DataFrame(rows)
    wins = sorted(R["window"].unique()) if len(R) else []
    if len(wins) < 40:
        raise RuntimeError(f"Muy pocas ventanas utilizables ({len(wins)}). Descarga más horas de historial.")
    cut = wins[int(len(wins) * train_frac)]
    R["set"] = np.where(R["window"] < cut, "train", "test")
    tr = R[R["set"] == "train"]

    # 1) ¿el score (tendencia/momentum) mejora al benchmark sin tendencia? (decidido con TRAIN)
    drift_supported = _brier(tr["p_raw"].to_numpy(), tr["y"].to_numpy()) < \
        _brier(tr["p_nd"].to_numpy(), tr["y"].to_numpy()) - 1.5e-3
    base_col = "p_raw" if drift_supported else "p_nd"

    # 2) ¿la calibración de Platt ayuda? se ajusta con 70 % de TRAIN y se comprueba con el 30 % final de TRAIN
    twins = sorted(tr["window"].unique())
    cut2 = twins[int(len(twins) * 0.7)]
    t_a, t_b = tr[tr["window"] < cut2], tr[tr["window"] >= cut2]
    a0, b0 = fit_platt(t_a[base_col].to_numpy(), t_a["y"].to_numpy())
    pb = 1 / (1 + np.exp(-(a0 * logit(t_b[base_col].to_numpy()) + b0)))
    platt_ok = _brier(pb, t_b["y"].to_numpy()) < _brier(t_b[base_col].to_numpy(), t_b["y"].to_numpy()) - 1e-4
    a, b = fit_platt(tr[base_col].to_numpy(), tr["y"].to_numpy()) if platt_ok else (1.0, 0.0)

    R["p_used"] = 1 / (1 + np.exp(-(a * logit(R[base_col].to_numpy()) + b))) if platt_ok else R[base_col]
    R["p_used"] = R["p_used"].clip(0.005, 0.995)

    # 3) Señales con la misma función de decisión que usa la app en vivo
    sigs, pnls, wins_l = [], [], []
    for _, r in R.iterrows():
        pm = float(r["p_nd"])                               # "precio justo" simulado del mercado
        hs = cfg.sim_half_spread
        q = {"yes_bid": max(pm - hs, 0.01), "yes_ask": min(pm + hs, 0.99),
             "no_bid": max(1 - pm - hs, 0.01), "no_ask": min(1 - pm + hs, 0.99)}
        unc, _, _ = uncertainty(r["z"], r["sd"], r["sratio"], 2 * hs, r["tau"], True, cfg)
        d = decide(float(r["p_used"]), q, float(r["tau"]), cfg, True, [], unc)
        if d["side"] is None:
            sigs.append("ESPERAR"); pnls.append(np.nan); wins_l.append(np.nan)
            continue
        s_ = d["sides"][d["side"]]
        won = (r["y"] == 1.0) if d["side"] == "UP" else (r["y"] == 0.0)
        sigs.append(d["signal"]); wins_l.append(1.0 if won else 0.0)
        pnls.append(s_["win"] if won else s_["loss"])
    R["signal"], R["pnl"], R["win"] = sigs, pnls, wins_l

    def summarize(sub: pd.DataFrame) -> dict:
        y = sub["y"].to_numpy()
        S = sub[sub["signal"] != "ESPERAR"]
        k, n = int(S["win"].sum()), len(S)
        lo, hi = wilson(k, n)
        return {
            "decisiones": len(sub), "ventanas": int(sub["window"].nunique()),
            "precision_direccional": float(np.mean((sub["p_raw"] >= 0.5) == (y == 1.0))) if len(sub) else float("nan"),
            "precision_persistencia": float(np.mean(sub["persist"] == y)) if len(sub) else float("nan"),
            "brier_modelo": _brier(sub["p_raw"].to_numpy(), y),
            "brier_usado": _brier(sub["p_used"].to_numpy(), y),
            "brier_sin_tendencia": _brier(sub["p_nd"].to_numpy(), y),
            "brier_constante_0.5": _brier(np.full(len(y), 0.5), y),
            "señales": n, "aciertos": k,
            "tasa_acierto": (k / n) if n else float("nan"), "ic95": (lo, hi),
            "pnl_total": float(S["pnl"].sum()) if n else 0.0,
            "pnl_medio": float(S["pnl"].mean()) if n else float("nan"),
        }

    test = R[R["set"] == "test"]
    sm_tr, sm_te = summarize(R[R["set"] == "train"]), summarize(test)

    # tabla de fiabilidad (test)
    bins = np.linspace(0, 1, 11)
    tb = test.assign(bin=pd.cut(test["p_used"], bins, include_lowest=True))
    rel = tb.groupby("bin", observed=True).agg(n=("y", "size"), p_media=("p_used", "mean"),
                                               frecuencia_real=("y", "mean")).reset_index()
    rel["bin"] = rel["bin"].astype(str)

    calibrated = bool(sm_te["decisiones"] >= 200
                      and sm_te["brier_usado"] < sm_te["brier_constante_0.5"] - 0.002
                      and sm_te["brier_usado"] <= sm_te["brier_sin_tendencia"] + 5e-4)
    return {"R": R, "train": sm_tr, "test": sm_te, "reliability": rel, "gaps": n_gaps,
            "validated": {"drift_supported": bool(drift_supported), "platt": (a, b),
                          "platt_ok": bool(platt_ok), "calibrated": calibrated},
            "periodo": (times[0], times[-1])}


# ──────────────────────────────────────────────────────────────────────────────
# ESTADO, HISTORIAL Y UTILIDADES DE UI
# ──────────────────────────────────────────────────────────────────────────────
HIST_COLS = ["hora_utc", "contrato", "señal", "btc", "referencia", "yes_ask", "no_ask", "prob_modelo",
             "ventaja", "seg_restantes", "incertidumbre", "motivo", "bloqueo", "cierre_utc", "resultado", "acierto"]


def load_history() -> List[dict]:
    try:
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE).replace({np.nan: None}).to_dict("records")
    except Exception:
        pass
    return []


def save_history(hist: List[dict]) -> None:
    try:
        pd.DataFrame(hist, columns=HIST_COLS).to_csv(HISTORY_FILE, index=False)
    except Exception:
        pass


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("last_good", {})
    ss.setdefault("status", {})
    ss.setdefault("history", load_history())
    ss.setdefault("bt", None)
    ss.setdefault("wait_counts", {})
    ss.setdefault("wait_total", 0)


def safe_fetch(name: str, fn) -> Tuple[Optional[dict], bool]:
    """Devuelve (datos, falló). Si falla, conserva la última lectura buena (se marcará como antigua)."""
    ss = st.session_state
    try:
        out = fn()
        ss.last_good[name] = out
        ss.status[name] = {"ok": True, "error": None, "ok_at": out["fetched_at"], "latency_ms": out.get("latency_ms")}
        return out, False
    except Exception as e:  # noqa: BLE001
        prev = ss.status.get(name, {})
        ss.status[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}",
                           "ok_at": prev.get("ok_at"), "latency_ms": None}
        return ss.last_good.get(name), True


def loc(dt: Optional[datetime], fmt: str = "%H:%M:%S") -> str:
    return "—" if dt is None else dt.astimezone(LOCAL_TZ).strftime(fmt)


def fmt_left(sec: float) -> str:
    sec = max(int(sec), 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


CSS = """
<style>
.stApp {background: radial-gradient(900px 500px at 10% -10%, #2a1a52 0%, #0e0b1a 55%) fixed;}
.block-container {padding: .7rem .65rem 3rem !important; max-width: 640px !important;}
section[data-testid="stSidebar"] {background:#120e24;border-right:1px solid #2b2150;}
h1,h2,h3,h4,h5 {color:#e9e5f5;}
h3 {font-size:1.25rem !important;} h4,h5 {font-size:1.02rem !important;}
/* columnas SIEMPRE lado a lado (en celular Streamlit las apila por defecto) */
[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .45rem !important;}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="column"] {min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;}
div[data-testid="stMetric"] {background:#1a1530;border:1px solid #2f2459;border-radius:12px;padding:7px 10px;}
div[data-testid="stMetricValue"] {color:#c4b5fd; font-size:1.18rem;}
div[data-testid="stMetricLabel"] p {font-size:.72rem;}
button, [role="tab"] {min-height:44px;}
[role="tab"] {font-size:.85rem; padding:0 .6rem;}
.stButton>button[kind="primary"] {background:#7c3aed;border:0;}
.sig {border:2px solid;border-radius:18px;padding:14px 16px;background:#161129;text-align:center;
      box-shadow:0 0 26px rgba(124,58,237,.25);}
.sig-label {color:#a78bfa;letter-spacing:.2em;font-size:.72rem;}
.sig-main {font-size:2.3rem;font-weight:800;line-height:1.1;}
.sig-sub {color:#cfc8ea;font-size:.85rem;margin-top:4px;word-break:break-all;}
.why {margin:.55rem 0 .2rem;padding:.55rem .7rem;border-radius:12px;background:#2a1d0a;border:1px solid #7a4a0a;
      color:#fde7bd;font-size:.85rem;line-height:1.35;}
.note {margin:.4rem 0;padding:.5rem .7rem;border-radius:12px;background:#12203a;border:1px solid #28508f;
       color:#cfe2ff;font-size:.83rem;line-height:1.35;}
.pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;margin-right:6px;}
.ok {background:#14532d;color:#bbf7d0;} .bad {background:#7f1d1d;color:#fecaca;} .old {background:#78350f;color:#fde68a;}
.frow {border-left:4px solid #444;background:#1a1530;border-radius:8px;padding:.35rem .6rem;margin:.28rem 0;}
.frow.ok {border-color:#22c55e;background:#10231a;color:inherit;}
.frow.fail {border-color:#ef4444;background:#2b1216;color:inherit;}
.frow.warn {border-color:#f59e0b;background:#2a2110;color:inherit;}
.frow.na {border-color:#555;background:#1a1726;color:inherit;opacity:.8;}
.fh {font-weight:700;font-size:.86rem;color:#ece8fa;}
.fm {font-size:.8rem;color:#cfc8ea;margin-top:1px;}
.fm .fv {font-weight:600;color:#fff;margin-right:.6rem;}
.fm .fl {color:#9a90c4;}
.fd {font-size:.78rem;color:#f3d9d9;margin-top:3px;line-height:1.3;}
table.kv, table.tb {width:100%;border-collapse:collapse;font-size:.8rem;margin:.2rem 0 .5rem;}
table.kv td {padding:3px 4px;border-bottom:1px solid #2b2150;vertical-align:top;}
table.kv td.k {color:#a78bfa;width:52%;} table.kv td.v {text-align:right;color:#f1edfb;word-break:break-word;}
table.tb th {color:#a78bfa;font-weight:600;text-align:right;padding:3px 4px;border-bottom:1px solid #3a2d70;}
table.tb th:first-child, table.tb td:first-child {text-align:left;}
table.tb td {padding:3px 4px;border-bottom:1px solid #2b2150;text-align:right;color:#f1edfb;}
table.tb td:first-child {color:#a78bfa;}
</style>
"""


def signal_card(signal: str, subtitle: str) -> str:
    color = {"BUY UP": "#22c55e", "BUY DOWN": "#ef4444", "ESPERAR": "#f59e0b"}.get(signal, "#a78bfa")
    return (f'<div class="sig" style="border-color:{color}"><div class="sig-label">SEÑAL</div>'
            f'<div class="sig-main" style="color:{color}">{esc(signal)}</div>'
            f'<div class="sig-sub">{esc(subtitle)}</div></div>')


def kv_html(rows: List[Tuple[str, Any]]) -> str:
    body = "".join(f"<tr><td class='k'>{esc(k)}</td><td class='v'>{esc(v)}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{body}</table>"


def table_html(headers: List[str], rows: List[List[Any]]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='tb'><tr>{th}</tr>{body}</table>"


ICON = {"ok": "✅", "fail": "⛔", "warn": "⚠️", "na": "➖"}


def filters_html(filters: List[dict]) -> str:
    out = []
    for f in filters:
        cls = f["status"]
        val = f'<span class="fv">{esc(f["value"])}</span>' if f["value"] else ""
        lim = f'<span class="fl">límite: {esc(f["limit"])}</span>' if f["limit"] else ""
        mid = f'<div class="fm">{val}{lim}</div>' if (val or lim) else ""
        det = f'<div class="fd">{esc(f["detail"])}</div>' if f["detail"] and cls in ("fail", "warn") else ""
        tag = "" if f["blocking"] or cls == "ok" else " (informativo)"
        out.append(f'<div class="frow {cls}"><div class="fh">{ICON[cls]} {esc(f["name"])}{tag}</div>{mid}{det}</div>')
    return "".join(out)


def make_chart(f: pd.DataFrame, forming: Optional[pd.Series], ctr: Optional[Contract], minutes: int) -> go.Figure:
    d = f.tail(minutes).copy()
    tz = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    x = tz(d["time"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.55, 0.12, 0.16, 0.17])
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="BTC-USD 1m", increasing_line_color="#22c55e",
                                 decreasing_line_color="#ef4444"), row=1, col=1)
    if forming is not None:
        fx = pd.Timestamp(forming["time"]).tz_convert(LOCAL_TZ).tz_localize(None)
        fig.add_trace(go.Candlestick(x=[fx], open=[forming["open"]], high=[forming["high"]], low=[forming["low"]],
                                     close=[forming["close"]], name="Vela actual (formándose)",
                                     increasing_line_color="#e9d5ff", decreasing_line_color="#e9d5ff",
                                     increasing_fillcolor="rgba(0,0,0,0)", decreasing_fillcolor="rgba(0,0,0,0)"),
                      row=1, col=1)
    for n, col in ((9, "#f0abfc"), (21, "#8b5cf6"), (50, "#38bdf8")):
        fig.add_trace(go.Scatter(x=x, y=d[f"ema{n}"], name=f"EMA {n}", line=dict(color=col, width=1.4)), row=1, col=1)
    if ctr is not None and ctr.reference is not None:
        fig.add_hline(y=ctr.reference, line=dict(color="#facc15", dash="dash", width=1.2), row=1, col=1,
                      annotation_text="Referencia", annotation_position="top left")
    fig.add_trace(go.Bar(x=x, y=d["volume"], name="Volumen", marker_color="#6d28d9"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["rsi"], name="RSI 14", line=dict(color="#c4b5fd")), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ef4444", dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#22c55e", dash="dot", width=1), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=d["macd_hist"], name="MACD hist", marker_color="#7c3aed"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["macd"], name="MACD", line=dict(color="#f0abfc", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["macd_signal"], name="Señal", line=dict(color="#38bdf8", width=1.2)), row=4, col=1)
    if ctr is not None:
        for t0 in (ctr.window_start, ctr.close_time):
            tx = pd.Timestamp(t0).tz_convert(LOCAL_TZ).tz_localize(None)
            if x.iloc[0] <= tx <= x.iloc[-1] + pd.Timedelta(minutes=16):
                fig.add_shape(type="line", x0=tx, x1=tx, y0=0, y1=1, xref="x", yref="paper",
                              line=dict(color="#a78bfa", dash="dot", width=1))
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=2, r=2, t=6, b=2), dragmode=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#120e24",
                      xaxis_rangeslider_visible=False, showlegend=False, font=dict(size=10))
    return fig


def factor_lines(row: pd.Series, direction: int) -> Tuple[List[str], List[str]]:
    pro, con = [], []
    for label, col, _ in GROUPS:
        v = float(row[col])
        if abs(v) < 0.15:
            continue
        (pro if np.sign(v) == direction else con).append(f"{label}: {v:+.2f}")
    return pro, con


# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS EN VIVO (calcula todo y devuelve un diccionario; sin dibujar nada)
# ──────────────────────────────────────────────────────────────────────────────
def analyze(cfg: Cfg, forced: Optional[str]) -> dict:
    now = utcnow()
    tick, tick_fail = safe_fetch("Coinbase · precio", fetch_ticker)
    cand, cand_fail = safe_fetch("Coinbase · velas 1m", fetch_candles)
    kal, kal_fail = safe_fetch("Kalshi · mercados", fetch_kalshi_markets)
    stat = st.session_state.status
    F: List[dict] = []

    def add(fid, name, state, value="", limit="", detail="", blocking=True):
        F.append(mkf(fid, name, state, value, limit, detail, blocking))

    def age_of(o: Optional[dict]) -> Optional[float]:
        return None if o is None else (now - o["fetched_at"]).total_seconds()

    def err_of(name: str) -> str:
        e = (stat.get(name) or {}).get("error")
        return f" Error: {e}" if e else ""

    # ── 1. Precio de Coinbase ──
    a = age_of(tick)
    lim_age = f"≤ {cfg.max_age_s} s"
    if tick is None:
        add("price", "Precio BTC (Coinbase)", "fail", "sin datos", lim_age,
            "Precio de Coinbase no disponible o desactualizado." + err_of("Coinbase · precio"))
    elif tick_fail or a > cfg.max_age_s:
        add("price", "Precio BTC (Coinbase)", "fail", f"dato de hace {a:.0f} s", lim_age,
            "Precio de Coinbase no disponible o desactualizado." + err_of("Coinbase · precio"))
    else:
        add("price", "Precio BTC (Coinbase)", "ok", f"{fnum(tick['price'], 2, '$')} · hace {a:.1f} s", lim_age)

    # ── 2. Hora de la última operación de Coinbase ──
    ex = None if tick is None else tick.get("exchange_time")
    if ex is None:
        add("trade_time", "Hora de la última operación", "warn", "sin hora", f"≤ {cfg.max_trade_age_s} s",
            "Coinbase no informó la hora de la última operación (no bloquea).", blocking=False)
    else:
        ta = (now - ex).total_seconds()
        if ta > cfg.max_trade_age_s:
            add("trade_time", "Hora de la última operación", "fail", f"{loc(ex)} (hace {ta:.0f} s)",
                f"≤ {cfg.max_trade_age_s} s",
                f"La última operación de Coinbase es de hace {ta:.0f} s: el precio podría estar congelado.")
        else:
            add("trade_time", "Hora de la última operación", "ok", f"{loc(ex)} (hace {max(ta, 0):.0f} s)",
                f"≤ {cfg.max_trade_age_s} s")

    # ── 3. Velas e indicadores ──
    closed = f = forming = row = None
    n_gap = 0
    lag = None
    if cand is not None:
        df = cand["df"]
        end_closed = df["time"] + pd.Timedelta(seconds=60) <= pd.Timestamp(now)
        closed_raw, fm = df[end_closed], df[~end_closed]
        if len(closed_raw):
            closed, n_gap = fill_gaps(closed_raw.reset_index(drop=True))
            last_end = closed["time"].iloc[-1].to_pydatetime() + timedelta(seconds=60)
            lag = (now - last_end).total_seconds()
            f = compute_features(closed)
            row = f.iloc[-1]
        if len(fm):
            forming = fm.iloc[-1]
    lim_c = f"retraso ≤ {CANDLE_MAX_LAG_S} s"
    if cand is None or cand_fail:
        add("candles", "Velas de 1 min (Coinbase)", "fail", "sin datos", lim_c,
            "Velas de Coinbase no disponibles o desactualizadas." + err_of("Coinbase · velas 1m"))
    elif closed is None:
        add("candles", "Velas de 1 min (Coinbase)", "fail", "0 velas cerradas", lim_c,
            "Coinbase no devolvió ninguna vela cerrada.")
    elif lag > CANDLE_MAX_LAG_S:
        add("candles", "Velas de 1 min (Coinbase)", "fail", f"última vela cerró hace {lag:.0f} s", lim_c,
            "La última vela cerrada de Coinbase está atrasada (> 2.5 min).")
    else:
        add("candles", "Velas de 1 min (Coinbase)", "ok",
            f"{len(closed)} velas · última cerró hace {max(lag, 0):.0f} s", lim_c)
    if f is None:
        add("warmup", "Indicadores calculados", "na", blocking=False)
    else:
        w_ok = len(f) >= WARMUP and bool(row["valid"])
        add("warmup", "Indicadores calculados", "ok" if w_ok else "fail", f"{len(f)} velas", f"≥ {WARMUP} y sin NaN",
            "" if w_ok else f"Datos insuficientes para los indicadores (se necesitan ≥ {WARMUP} velas cerradas válidas).")
    add("forming", "Vela actual de Coinbase", "ok" if forming is not None else "warn",
        "presente" if forming is not None else "ausente", "",
        "Coinbase aún no publicó la vela del minuto en curso (pasa en los primeros segundos de cada minuto). "
        "Solo afecta a la gráfica; ya NO bloquea la señal.", blocking=False)

    # ── 4. Kalshi: frescura ──
    a_k = age_of(kal)
    if kal is None:
        add("kalshi", "Datos de Kalshi", "fail", "sin datos", lim_age,
            "Datos de Kalshi no disponibles o desactualizados." + err_of("Kalshi · mercados"))
    elif kal_fail or a_k > cfg.max_age_s:
        add("kalshi", "Datos de Kalshi", "fail", f"dato de hace {a_k:.0f} s", lim_age,
            "Datos de Kalshi no disponibles o desactualizados." + err_of("Kalshi · mercados"))
    else:
        add("kalshi", "Datos de Kalshi", "ok", f"descargados hace {a_k:.1f} s", lim_age)

    # ── 5. Reloj local ──
    signed = []
    for o in (tick, cand, kal):
        sd_ = o.get("server_date") if o else None
        if sd_ is not None:
            signed.append((o["fetched_at"] - sd_).total_seconds())
    if signed:
        skew = min(signed)   # los sesgos (latencia, cachés) son siempre positivos: el mínimo es el más fiel
        sk_ok = abs(skew) <= cfg.max_clock_skew_s
        add("clock", "Reloj del dispositivo/servidor", "ok" if sk_ok else "fail", f"desfase ≈ {skew:+.1f} s",
            f"±{cfg.max_clock_skew_s:.0f} s",
            "" if sk_ok else f"El reloj de la máquina que ejecuta la app difiere ~{abs(skew):.0f} s de los servidores; sincronízalo.")
    else:
        add("clock", "Reloj del dispositivo/servidor", "warn", "sin cabecera Date", "",
            "Los servidores no enviaron hora: no se pudo comprobar (no bloquea).", blocking=False)

    # ── 6. Contrato ──
    mk = None
    ctr: Optional[Contract] = None
    open_markets: List[dict] = []
    if kal is not None:
        open_markets = sorted([m for m in kal["markets"] if m.get("close_time")], key=lambda m: m["close_time"])
        mk = select_market(kal["markets"], now, forced)
    if kal is None:
        add("contract", "Contrato analizado", "na", blocking=False)
    elif mk is None:
        add("contract", "Contrato analizado", "fail", "ninguno", "", "No hay contratos KXBTC15M abiertos en este momento.")
    else:
        add("contract", "Contrato analizado", "ok", f"{mk['ticker']} · cierra {loc(mk['close_time'])}", "")
    if mk is not None:
        if forced and mk["ticker"] != forced:
            add("contract_match", "Es el contrato que elegiste", "fail", f"analiza {mk['ticker']}", f"= {forced}",
                f"El contrato elegido ({forced}) ya no está abierto; se analizaría {mk['ticker']}, que podría no ser "
                "el que estás viendo. Elige otro en Configuración o cambia a Automático.")
        elif forced:
            add("contract_match", "Es el contrato que elegiste", "ok", "coincide", f"= {forced}")
        else:
            add("contract_match", "Es el contrato que elegiste", "ok", "modo Automático (el que corre ahora)", "",
                "Compara el ticker de arriba con el de tu pantalla de Kalshi.")
    else:
        add("contract_match", "Es el contrato que elegiste", "na", blocking=False)

    if mk is not None and closed is not None:
        ctr = build_contract(mk, closed, now)
    if ctr is None:
        add("rules", "Reglas del contrato soportadas", "na", blocking=False)
        add("reference", "Referencia / umbral", "na", blocking=False)
    else:
        add("rules", "Reglas del contrato soportadas", "ok" if ctr.supported else "fail",
            f"strike_type={mk['strike_type']!r}" + (" · YES si ≥ ref." if ctr.yes_if_above else " · YES si < ref."), "",
            "" if ctr.supported else "Reglas del contrato no soportadas por el analizador. " + " ".join(ctr.notes))
        add("reference", "Referencia / umbral", "ok" if ctr.reference is not None else "fail",
            fnum(ctr.reference, 2, "$") + f" ({ctr.reference_kind})", "",
            "" if ctr.reference is not None else "No se pudo determinar el precio de referencia/umbral del contrato. "
            + " ".join(ctr.notes))

    S0: Optional[float] = None
    if tick is not None:
        S0 = tick["price"]
    elif row is not None:
        S0 = float(row["close"])

    if ctr is not None and ctr.reference and S0:
        dev = abs(math.log(S0 / ctr.reference))
        d_ok = dev <= cfg.max_ref_dev
        add("ref_sane", "Referencia coherente con el precio", "ok" if d_ok else "fail", f"{dev * 100:.2f}% de distancia",
            f"≤ {cfg.max_ref_dev * 100:.0f}%",
            "" if d_ok else f"El precio de BTC ({fnum(S0, 2, '$')}) está {dev * 100:.1f}% lejos de la referencia "
            f"({fnum(ctr.reference, 2, '$')}): en 15 min eso es casi imposible, parece un error de escala o de contrato.")
    else:
        add("ref_sane", "Referencia coherente con el precio", "na", blocking=False)

    # ── 7. Coherencia de precios de Kalshi ──
    q = {k: None for k in PRICE_KEYS}
    if mk is not None:
        q = {k: mk[k] for k in PRICE_KEYS}
        qs, qd = quote_check(q, cfg.quote_tol)
        add("quote_check", "Precios YES/NO coherentes", qs, f"YES {cents(q['yes_bid'])}/{cents(q['yes_ask'])} · "
            f"NO {cents(q['no_bid'])}/{cents(q['no_ask'])}", f"suma = 100¢ ±{cfg.quote_tol * 100:.0f}¢",
            "Precios inconsistentes (posible error al interpretarlos): " + qd if qd else "")
    else:
        add("quote_check", "Precios YES/NO coherentes", "na", blocking=False)

    # ── 8. Modelo ──
    tau_min = None if ctr is None else (ctr.close_time - now).total_seconds() / 60.0
    mp = None
    p_final = p_model_yes = p_nd_yes = None
    unc, why_unc, unc_pts, calibrated, notes_model = "Alta", [], 0, False, []
    platt_used = None
    if (row is not None and ctr is not None and ctr.reference and S0 and tau_min is not None
            and tau_min > 0 and bool(row["valid"])):
        bt = st.session_state.bt["validated"] if (cfg.use_backtest and st.session_state.bt) else None
        use_drift, platt = True, None
        if bt:
            use_drift = bt["drift_supported"]
            platt = bt["platt"] if bt["platt_ok"] else None
            calibrated = bt["calibrated"]
            notes_model.append("Modelo ajustado según el último backtest" +
                               ("" if use_drift else " (el score no mejoró al benchmark: se usa SIN tendencia)") + ".")
        else:
            notes_model.append("Sin backtest ejecutado en esta sesión: probabilidad NO calibrada.")
        mp = model_prob(S0, ctr.reference, tau_min, float(row["sigma"]), float(row["score"]) if use_drift else 0.0, cfg)
        p_model_yes = mp["p"] if ctr.yes_if_above else 1 - mp["p"]
        p_nd_yes = mp["p_nodrift"] if ctr.yes_if_above else 1 - mp["p_nodrift"]
        p_final = p_model_yes
        if platt:
            p_final = apply_platt(p_final, *platt)
            platt_used = platt
        spreads = [x - y for x, y in ((q["yes_ask"], q["yes_bid"]), (q["no_ask"], q["no_bid"]))
                   if x is not None and y is not None]
        unc, why_unc, unc_pts = uncertainty(mp["z"], mp["sd"], float(row["sigma_ratio"]),
                                            min(spreads) if spreads else None, tau_min, calibrated, cfg)

    implied = None
    if q["yes_bid"] is not None and q["yes_ask"] is not None:
        implied = (q["yes_bid"] + q["yes_ask"]) / 2
    if p_final is not None and implied is not None:
        diff = p_final - implied
        big = abs(diff) > 0.30
        add("divergence", "Modelo vs mercado", "warn" if big else "ok", f"{diff * 100:+.0f} pts", "≤ 30 pts",
            "El modelo y Kalshi discrepan más de 30 puntos: verifica que el contrato y la referencia sean los que "
            "ves en Kalshi y que Coinbase no se separe del BRTI (no bloquea).", blocking=False)
    else:
        add("divergence", "Modelo vs mercado", "na", blocking=False)

    # ── 9. Decisión ──
    problems = [f_["detail"] for f_ in F if f_["status"] == "fail" and f_["blocking"] and f_["detail"]]
    if p_final is None:
        dec = {"signal": "ESPERAR", "side": None, "best": None, "sides": {}, "req_edge": None,
               "reasons": problems or ["Sin datos suficientes para estimar."],
               "filters": [mkf(i, n, "na", blocking=False) for i, n in DECISION_FILTERS]}
    else:
        dec = decide(p_final, q, tau_min, cfg, calibrated, problems, unc, "; ".join(why_unc))
    all_filters = F + dec["filters"]
    blockers = [f_ for f_ in all_filters if f_["status"] == "fail" and f_["blocking"]]
    return {"now": now, "cfg": cfg, "forced": forced, "tick": tick, "cand": cand, "kal": kal, "mk": mk, "ctr": ctr,
            "open_markets": open_markets, "S0": S0, "tau_min": tau_min, "f": f, "closed": closed, "forming": forming,
            "row": row, "n_gap": n_gap, "lag": lag, "mp": mp, "p_final": p_final, "p_model_yes": p_model_yes,
            "p_nd_yes": p_nd_yes, "platt": platt_used, "unc": unc, "why_unc": why_unc, "unc_pts": unc_pts,
            "calibrated": calibrated, "notes_model": notes_model, "q": q, "implied": implied, "dec": dec,
            "signal": dec["signal"], "filters": all_filters, "blockers": blockers}


# ──────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO: tablas y texto
# ──────────────────────────────────────────────────────────────────────────────
def reading_rows(R: dict) -> Dict[str, List[Tuple[str, str]]]:
    """Valores reales que está leyendo el programa, agrupados."""
    mp, ctr, row, S0 = R["mp"], R["ctr"], R["row"], R["S0"]
    tick, mk = R["tick"], R["mk"]
    probs: List[Tuple[str, str]] = []
    if R["p_final"] is not None and mp is not None:
        probs += [
            ("Prob. UP (compra YES) — leída", pct(R["p_final"])),
            ("Prob. DOWN (compra NO) — leída", pct(1 - R["p_final"])),
            ("Prob. del modelo antes de calibrar", pct(R["p_model_yes"])),
            ("Prob. benchmark sin tendencia", pct(R["p_nd_yes"])),
            ("Calibración Platt (a, b)", "no aplicada" if R["platt"] is None else f"a={R['platt'][0]:.2f}, b={R['platt'][1]:+.2f}"),
            ("Prob. implícita Kalshi (YES, punto medio)", pct(R["implied"])),
            ("Diferencia modelo − mercado", "—" if R["implied"] is None else f"{(R['p_final'] - R['implied']) * 100:+.1f} pts"),
            ("Último precio operado YES", "—" if mk is None else cents(mk.get("last_price"))),
            ("Modelo calibrado", "Sí" if R["calibrated"] else "No (+2¢ de margen y +1 pt de incertidumbre)"),
        ]
    else:
        probs.append(("Probabilidad", "no calculada (faltan datos: mira los filtros ⛔)"))
    read: List[Tuple[str, str]] = []
    if S0 is not None:
        read.append(("BTC/USD (Coinbase)", fnum(S0, 2, "$")))
    if ctr is not None and ctr.reference is not None:
        read.append(("Referencia / umbral", fnum(ctr.reference, 2, "$")))
        if S0 is not None:
            read.append(("Precio − referencia", f"{S0 - ctr.reference:+,.2f} USD"))
    if mp is not None:
        read += [("Distancia al umbral", f"{mp['dist'] * 1e4:+.1f} pb"),
                 ("Desviación esperada (sd)", f"{mp['sd'] * 1e4:.1f} pb"),
                 ("z = distancia/sd (+ tendencia)", f"{mp['z']:+.2f}"),
                 ("Aporte de la tendencia (z)", f"{mp['drift_z']:+.3f}"),
                 ("Tiempo efectivo del modelo", f"{mp['t_eff']:.2f} min")]
    if R["tau_min"] is not None:
        read.append(("Tiempo restante", f"{R['tau_min'] * 60:.0f} s"))
    ind: List[Tuple[str, str]] = []
    if row is not None:
        ind = [("RSI 14", fnum(row["rsi"], 1)), ("MACD", fnum(row["macd"], 2)), ("MACD señal", fnum(row["macd_signal"], 2)),
               ("MACD histograma", fnum(row["macd_hist"], 2)), ("ATR 14", fnum(row["atr"], 2, "$")),
               ("EMA 9 / 21 / 50", f"{fnum(row['ema9'], 0)} / {fnum(row['ema21'], 0)} / {fnum(row['ema50'], 0)}"),
               ("σ por minuto", f"{row['sigma'] * 1e4:.1f} pb"), ("σ reciente / σ 60 min", fnum(row["sigma_ratio"], 2)),
               ("Volumen vs media 20", "—" if pd.isna(row["vol_ratio"]) else f"{row['vol_ratio']:.2f}×"),
               ("Momentum 3 / 5 velas", f"{row['mom3'] * 1e4:+.1f} / {row['mom5'] * 1e4:+.1f} pb")]
        ind += [(lab.split(" (")[0], f"{float(row[col]):+.2f}") for lab, col, _ in GROUPS]
        ind.append(("Score total (−1…+1)", f"{float(row['score']):+.2f}"))
    times: List[Tuple[str, str]] = [
        ("Actualizado (local)", loc(R["now"], "%Y-%m-%d %H:%M:%S")), ("Actualizado (UTC)", R["now"].strftime("%H:%M:%S"))]
    if tick is not None:
        times += [("Precio descargado", loc(tick["fetched_at"])),
                  ("Última operación Coinbase", loc(tick.get("exchange_time"))),
                  ("Latencia Coinbase precio", fnum(tick.get("latency_ms"), 0, "", " ms"))]
    if R["closed"] is not None:
        last_open = R["closed"]["time"].iloc[-1].to_pydatetime()
        times.append(("Última vela cerrada (inicio)", loc(last_open)))
    if R["kal"] is not None:
        times.append(("Kalshi descargado", loc(R["kal"]["fetched_at"])))
    if mk is not None:
        times += [("Kalshi: updated_time", loc(mk.get("updated"))),
                  ("Ventana: inicio", loc(ctr.window_start if ctr else None)), ("Ventana: cierre", loc(mk["close_time"]))]
    return {"probs": probs, "read": read, "ind": ind, "times": times}


def sides_html(R: dict, cfg: Cfg) -> str:
    dec, mk = R["dec"], R["mk"]
    S = dec.get("sides") or {}
    if not S:
        return ""
    best = dec.get("best")

    def g(side: str, fn: Callable[[dict], str]) -> str:
        s = S.get(side)
        return "—" if not s else fn(s)

    rows = [
        ["Prob. modelo"] + [g(k, lambda s: pct(s["p"])) for k in ("UP", "DOWN")],
        ["Compra (ask)"] + [g(k, lambda s: cents(s["ask"])) for k in ("UP", "DOWN")],
        ["Venta (bid)"] + [g(k, lambda s: cents(s["bid"])) for k in ("UP", "DOWN")],
        ["Spread"] + [g(k, lambda s: cents(s["spread"])) for k in ("UP", "DOWN")],
        ["Comisión"] + [g(k, lambda s: f"{s['fee'] * 100:.2f}¢") for k in ("UP", "DOWN")],
        ["Deslizamiento"] + [g(k, lambda s: cents(cfg.slippage)) for k in ("UP", "DOWN")],
        ["Costo equilibrio"] + [g(k, lambda s: pct(s["breakeven"])) for k in ("UP", "DOWN")],
        ["Prob. mínima p/ señal"] + [g(k, lambda s: pct(s["req_p"]) + (" ⛔" if s["req_p"] > 0.995 else ""))
                                     for k in ("UP", "DOWN")],
        ["Ventaja neta"] + [g(k, lambda s: f"{s['edge'] * 100:+.1f}¢") for k in ("UP", "DOWN")],
        ["Ventaja exigida", f"{dec['req_edge'] * 100:.1f}¢", f"{dec['req_edge'] * 100:.1f}¢"],
    ]
    if mk is not None:
        rows.insert(3, ["Profundidad", fnum(mk.get("yes_ask_size"), 0) if S.get("UP") else "—",
                        fnum(mk.get("yes_bid_size"), 0) if S.get("DOWN") else "—"])
    hdr = ["", "UP (YES)" + (" ★" if best == "UP" else ""), "DOWN (NO)" + (" ★" if best == "DOWN" else "")]
    return table_html(hdr, rows)


def edge_explanation(R: dict, cfg: Cfg) -> str:
    dec = R["dec"]
    if not dec.get("best") or dec.get("req_edge") is None:
        return ""
    # Se explican AMBOS lados que no alcanzan la ventaja exigida, primero el de mayor probabilidad
    # (el que normalmente ves como "80-90 %" en pantalla).
    sides = sorted(((k, v) for k, v in dec["sides"].items() if v), key=lambda kv: -kv[1]["p"])
    parts, high = [], False
    for b, s in sides:
        if s["edge"] >= dec["req_edge"]:
            continue
        parts.append(f"Lado {b}: el modelo da {pct(s['p'])}, pero comprarlo cuesta {cents(s['ask'])} más comisión "
                     f"{s['fee'] * 100:.2f}¢ y deslizamiento {cents(cfg.slippage)} → equilibrio {pct(s['breakeven'])}. "
                     f"Con la ventaja exigida ({dec['req_edge'] * 100:.1f}¢) tendría que dar ≥ {pct(s['req_p'])}.")
        high = high or s["p"] >= 0.75
    if not parts:
        return ""
    txt = " ".join(parts)
    if high:
        txt += (" Una probabilidad alta NO es una ventaja por sí sola: si el mercado ya la cobra a ese precio, "
                "no hay nada que ganar y el filtro hace bien en decir ESPERAR.")
    if not R["calibrated"]:
        txt += (f" Como el modelo no está validado, se suman {cfg.uncal_extra * 100:.0f}¢ de margen "
                "(ejecuta el Backtest para validarlo; no se activa solo).")
    return txt


def build_diag_text(R: dict, cfg: Cfg) -> str:
    L = [f"DIAGNÓSTICO {loc(R['now'], '%Y-%m-%d %H:%M:%S')} local / {R['now'].strftime('%H:%M:%S')} UTC",
         f"Señal: {R['signal']}",
         f"Contrato: {R['ctr'].ticker if R['ctr'] else '—'}  cierre: {loc(R['mk']['close_time']) if R['mk'] else '—'}",
         "", "FILTROS"]
    for f_ in R["filters"]:
        tag = {"ok": "OK   ", "fail": "FALLA", "warn": "AVISO", "na": "N/D  "}[f_["status"]]
        L.append(f"[{tag}] {f_['name']}: {f_['value']}" + (f" | límite {f_['limit']}" if f_["limit"] else "")
                 + (f" | {f_['detail']}" if f_["detail"] and f_["status"] in ("fail", "warn") else ""))
    rr = reading_rows(R)
    for title, key in (("PROBABILIDADES", "probs"), ("LECTURAS", "read"), ("INDICADORES", "ind"), ("HORAS", "times")):
        L += ["", title] + [f"{k}: {v}" for k, v in rr[key]]
    mk = R["mk"]
    if mk is not None:
        L += ["", "KALSHI CRUDO", json.dumps(mk["raw"], ensure_ascii=False, default=str),
              f"derivados (no vinieron en la API): {mk['derived']}"]
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────
# VISTA EN VIVO
# ──────────────────────────────────────────────────────────────────────────────
def log_signal(rec: dict) -> None:
    hist: List[dict] = st.session_state.history
    last = next((h for h in reversed(hist) if h.get("contrato") == rec["contrato"]), None)
    now = utcnow()
    changed = last is None or last.get("señal") != rec["señal"]
    if not changed and rec["señal"] == "ESPERAR" and (last.get("bloqueo") or "") != (rec.get("bloqueo") or ""):
        changed = True   # sigue en ESPERAR pero cambió el motivo principal
    heartbeat = False
    if last is not None and rec["señal"] != "ESPERAR":
        try:
            heartbeat = (now - pd.to_datetime(last["hora_utc"], utc=True).to_pydatetime()).total_seconds() >= 90
        except Exception:
            heartbeat = True
    if changed or heartbeat:
        hist.append(rec)
        save_history(hist)


def resolve_history(now: datetime) -> None:
    """Rellena el resultado real (yes/no) de señales BUY ya vencidas, consultando a Kalshi."""
    hist: List[dict] = st.session_state.history
    pending = [h for h in hist if str(h.get("señal", "")).startswith("BUY") and not h.get("resultado")]
    done = 0
    cache: Dict[str, Optional[str]] = {}
    for h in pending:
        try:
            close = pd.to_datetime(h["cierre_utc"], utc=True).to_pydatetime()
        except Exception:
            continue
        if now < close + timedelta(seconds=90) or done >= 2:
            continue
        t = h["contrato"]
        if t not in cache:
            try:
                cache[t] = fetch_kalshi_market_result(t)
            except Exception:
                cache[t] = None
            done += 1
        res = cache[t]
        if res:
            h["resultado"] = res
            h["acierto"] = bool((h["señal"] == "BUY UP" and res == "yes") or (h["señal"] == "BUY DOWN" and res == "no"))
    if done:
        save_history(hist)


def status_panel(now: datetime) -> None:
    st.markdown("##### Conexiones")
    rows = []
    for name in ("Coinbase · precio", "Coinbase · velas 1m", "Kalshi · mercados"):
        s = st.session_state.status.get(name)
        if not s:
            rows.append(f"<div class='frow na'><div class='fh'>{esc(name)} <span class='pill old'>sin datos</span></div></div>")
            continue
        age = None if s["ok_at"] is None else (now - s["ok_at"]).total_seconds()
        if s["ok"]:
            rows.append(f"<div class='frow ok'><div class='fh'>{esc(name)} <span class='pill ok'>OK</span></div>"
                        f"<div class='fm'>{s['latency_ms']:.0f} ms · actualizado {loc(s['ok_at'])}</div></div>")
        else:
            extra = (f"último dato bueno ANTIGUO ({age:.0f} s): {loc(s['ok_at'])}" if age is not None else "sin datos previos")
            rows.append(f"<div class='frow fail'><div class='fh'>{esc(name)} <span class='pill bad'>ERROR</span></div>"
                        f"<div class='fm'>{esc(extra)}</div><div class='fd'>{esc(s['error'])}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_header(R: dict, cfg: Cfg) -> None:
    ctr, tau = R["ctr"], R["tau_min"]
    sub = f"{ctr.ticker} · quedan {fmt_left(tau * 60)}" if (ctr is not None and tau is not None) else "Sin contrato analizable"
    st.markdown(signal_card(R["signal"], sub), unsafe_allow_html=True)
    if R["signal"] == "ESPERAR" and R["blockers"]:
        b = R["blockers"][0]
        extra = f"  (+{len(R['blockers']) - 1} bloqueo(s) más)" if len(R["blockers"]) > 1 else ""
        st.markdown(f"<div class='why'>⛔ <b>{esc(b['name'])}</b>{esc(extra)}<br>{esc(b['detail'])}</div>",
                    unsafe_allow_html=True)
    now = R["now"]
    st.caption(f"Actualizado {loc(now)} ({LOCAL_TZ.key}) · {now.strftime('%H:%M:%S')} UTC · cada {cfg.refresh_s} s")


def render_metrics(R: dict) -> None:
    q, S0, tau = R["q"], R["S0"], R["tau_min"]
    c1, c2 = st.columns(2)
    c1.metric("BTC/USD (Coinbase)", "—" if S0 is None else f"${S0:,.2f}")
    c2.metric("Tiempo restante", "—" if tau is None else fmt_left(tau * 60))
    c3, c4 = st.columns(2)
    c3.metric("YES compra / venta", f"{cents(q['yes_ask'])} / {cents(q['yes_bid'])}")
    c4.metric("NO compra / venta", f"{cents(q['no_ask'])} / {cents(q['no_bid'])}")
    c5, c6 = st.columns(2)
    c5.metric("Prob. YES (modelo)", pct(R["p_final"]))
    c6.metric("Prob. YES (Kalshi)", pct(R["implied"]))


def render_diagnostics(R: dict, cfg: Cfg) -> None:
    sig = R["signal"]
    n_fail, n_warn = len(R["blockers"]), sum(1 for f_ in R["filters"] if f_["status"] == "warn")
    title = f"🔎 Diagnóstico · {n_fail} bloqueo(s) · {n_warn} aviso(s)"
    with st.expander(title, expanded=(sig == "ESPERAR")):
        ex = edge_explanation(R, cfg)
        if ex:
            st.markdown(f"<div class='note'>{esc(ex)}</div>", unsafe_allow_html=True)
        st.markdown("**Filtros (todos se evalúan a la vez)**")
        st.markdown(filters_html(R["filters"]), unsafe_allow_html=True)
        rr = reading_rows(R)
        st.markdown("**Probabilidad leída**")
        st.markdown(kv_html(rr["probs"]), unsafe_allow_html=True)
        sh = sides_html(R, cfg)
        if sh:
            st.markdown("**Costos y ventaja por lado** (★ = mejor opción)")
            st.markdown(sh, unsafe_allow_html=True)
        if rr["read"]:
            st.markdown("**Valores que lee el modelo**")
            st.markdown(kv_html(rr["read"]), unsafe_allow_html=True)
        if rr["ind"]:
            st.markdown("**Indicadores calculados** (última vela cerrada)")
            st.markdown(kv_html(rr["ind"]), unsafe_allow_html=True)
        if R["why_unc"]:
            st.markdown(f"**Incertidumbre: {R['unc']}** ({R['unc_pts']} pts · 'Alta' desde 4)")
            st.markdown("".join(f"<div class='fm'>• {esc(w)}</div>" for w in R["why_unc"]), unsafe_allow_html=True)
        st.markdown("**Horas**")
        st.markdown(kv_html(rr["times"]), unsafe_allow_html=True)
        mk = R["mk"]
        if mk is not None:
            st.markdown("**Valores crudos de Kalshi** (tal como llegan)")
            st.markdown(kv_html([(k, v) for k, v in mk["raw"].items()] + [("derivados por el programa", ", ".join(mk["derived"]) or "ninguno")]),
                        unsafe_allow_html=True)
        if R["open_markets"]:
            st.markdown("**Contratos abiertos que ve el programa** (compáralos con tu pantalla)")
            rows = [[("► " if (mk and m["ticker"] == mk["ticker"]) else "") + str(m["ticker"]).replace(SERIES + "-", ""),
                     loc(m["close_time"], "%H:%M"), fnum(m.get("strike"), 2, "$"), cents(m.get("yes_ask")), cents(m.get("no_ask"))]
                    for m in R["open_markets"][:6]]
            st.markdown(table_html(["Ticker", "Cierra", "Umbral", "YES", "NO"], rows), unsafe_allow_html=True)
        wc, wt = st.session_state.wait_counts, st.session_state.wait_total
        if wt:
            top = sorted(wc.items(), key=lambda kv: -kv[1])[:6]
            names = {f_["id"]: f_["name"] for f_ in R["filters"]}
            st.markdown(f"**Bloqueos más frecuentes en esta sesión** ({wt} lecturas en ESPERAR)")
            st.markdown(kv_html([(names.get(k, k), f"{v} ({v / wt * 100:.0f}%)") for k, v in top]), unsafe_allow_html=True)
        st.markdown("**Texto para copiar y pegar** (útil para pedir ayuda)")
        st.code(build_diag_text(R, cfg), language=None)


def render_explanation(R: dict, cfg: Cfg) -> None:
    mp, row, ctr, p_final = R["mp"], R["row"], R["ctr"], R["p_final"]
    if p_final is None or mp is None or row is None or ctr is None:
        return
    dec, S0 = R["dec"], R["S0"]
    with st.expander("Por qué (a favor / en contra)"):
        st.markdown(f"**Dirección estimada:** {'ALCISTA (YES)' if p_final >= 0.5 else 'BAJISTA (NO)'}  \n"
                    f"**Contrato:** {ctr.title} — referencia {fnum(ctr.reference, 2, '$')} ({ctr.reference_kind})")
        if dec["side"] is not None:
            dir_btc = 1 if (dec["side"] == "UP") == ctr.yes_if_above else -1
        else:
            dir_btc = 1 if mp["p"] >= 0.5 else -1
        pro, con = factor_lines(row, dir_btc)
        gap = S0 - ctr.reference
        (pro if np.sign(gap) == dir_btc else con).append(
            f"Precio vs referencia: {gap:+,.2f} USD ({mp['dist'] * 1e4:+.1f} pb; z = {mp['z']:+.2f})")
        st.markdown("**A favor**")
        st.write("\n".join(f"- {t}" for t in pro) or "- (ninguno destacable)")
        st.markdown("**En contra**")
        st.write("\n".join(f"- {t}" for t in con) or "- (ninguno destacable)")
        st.markdown("**Motivo de la señal:** " + " ".join(dec["reasons"]))
        for n_ in R["notes_model"]:
            st.caption(n_)
        best = dec.get("best")
        if best and dec["sides"].get(best):
            ask_ref = dec["sides"][best]["ask"]
            st.caption(f"Comisión de ejemplo para 10 contratos a {cents(ask_ref)}: "
                       f"${kalshi_fee_order(10, ask_ref, cfg.fee_rate):.2f} (se redondea al centavo por orden). "
                       f"Incluye deslizamiento supuesto de {cfg.slippage * 100:.0f}¢.")


def render_chart(R: dict, cfg: Cfg) -> None:
    f = R["f"]
    if f is None:
        return
    st.caption("Velas 1 min · EMA 9 rosa, 21 morada, 50 azul · abajo: volumen, RSI, MACD")
    st.plotly_chart(make_chart(f, R["forming"], R["ctr"], cfg.chart_minutes),
                    config={"displayModeBar": False, "staticPlot": True})


def render_contract(R: dict) -> None:
    mk, ctr = R["mk"], R["ctr"]
    if mk is None or ctr is None:
        return
    with st.expander("Datos y reglas del contrato de Kalshi"):
        st.markdown(kv_html([
            ("Ticker", ctr.ticker), ("Título", ctr.title), ("Estado (Kalshi)", mk.get("status")),
            ("Inicio de ventana", loc(ctr.window_start)), ("Cierre", loc(ctr.close_time)),
            ("Referencia", fnum(ctr.reference, 2, "$")), ("Tipo de referencia", ctr.reference_kind),
            ("YES gana si precio ≥ referencia", "sí" if ctr.yes_if_above else "no (invertido)"),
            ("Volumen", fnum(mk["volume"], 0)), ("Interés abierto", fnum(mk["open_interest"], 0)),
            ("Liquidez", fnum(mk["liquidity"], 2, "$"))]), unsafe_allow_html=True)
        for n_ in ctr.notes:
            st.caption("• " + n_)
        st.markdown("**Reglas oficiales (texto de Kalshi):**")
        st.write(ctr.rules or "(la API no devolvió texto de reglas)")


def record_wait(R: dict) -> None:
    if R["signal"] != "ESPERAR":
        return
    ss = st.session_state
    ss.wait_total += 1
    for b in R["blockers"]:
        ss.wait_counts[b["id"]] = ss.wait_counts.get(b["id"], 0) + 1


def render_live(cfg: Cfg, forced: Optional[str]) -> None:
    R = analyze(cfg, forced)
    record_wait(R)
    render_header(R, cfg)
    render_metrics(R)
    render_diagnostics(R, cfg)
    render_explanation(R, cfg)
    render_chart(R, cfg)
    status_panel(R["now"])
    render_contract(R)

    ctr, now, dec, q = R["ctr"], R["now"], R["dec"], R["q"]
    if ctr is not None:
        log_signal({"hora_utc": now.strftime("%Y-%m-%d %H:%M:%S"), "contrato": ctr.ticker, "señal": R["signal"],
                    "btc": None if R["S0"] is None else round(R["S0"], 2), "referencia": ctr.reference,
                    "yes_ask": q["yes_ask"], "no_ask": q["no_ask"],
                    "prob_modelo": None if R["p_final"] is None else round(R["p_final"], 4),
                    "ventaja": None if not dec.get("side") else round(dec["sides"][dec["side"]]["edge"], 4),
                    "seg_restantes": None if R["tau_min"] is None else round(R["tau_min"] * 60),
                    "incertidumbre": R["unc"], "motivo": " ".join(dec["reasons"])[:200],
                    "bloqueo": "+".join(b["id"] for b in R["blockers"]) if R["signal"] == "ESPERAR" else "",
                    "cierre_utc": ctr.close_time.strftime("%Y-%m-%d %H:%M:%S"), "resultado": None, "acierto": None})
    resolve_history(now)
    render_history()


def render_history() -> None:
    hist = st.session_state.history
    st.markdown("##### Historial de señales")
    if not hist:
        st.caption("Aún no hay señales registradas.")
        return
    h = pd.DataFrame(hist, columns=HIST_COLS)
    buys = h[h["señal"].astype(str).str.startswith("BUY")]
    res = buys[buys["resultado"].notna() & (buys["resultado"] != "")]
    k = int((res["acierto"].astype(str) == "True").sum())
    st.caption(f"Señales BUY: {len(buys)} · resueltas: {len(res)} · aciertos: {k}"
               + (f" ({k / len(res) * 100:.0f}%)" if len(res) else "")
               + "  — muestra pequeña: no es evidencia estadística.")
    v = h.iloc[::-1].head(60).copy()
    try:
        v["hora"] = pd.to_datetime(v["hora_utc"], utc=True).dt.tz_convert(LOCAL_TZ).dt.strftime("%H:%M:%S")
    except Exception:
        v["hora"] = v["hora_utc"]
    v["prob"] = v["prob_modelo"].apply(lambda x: "—" if x is None or pd.isna(x) else f"{float(x) * 100:.0f}%")
    v["ventaja"] = v["ventaja"].apply(lambda x: "—" if x is None or pd.isna(x) else f"{float(x) * 100:+.1f}¢")
    st.dataframe(v[["hora", "señal", "prob", "ventaja", "bloqueo", "resultado"]], hide_index=True)
    st.download_button("Descargar historial completo (CSV)", h.to_csv(index=False).encode("utf-8"),
                       file_name="signal_history.csv", mime="text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST (UI)
# ──────────────────────────────────────────────────────────────────────────────
def render_backtest(cfg: Cfg) -> None:
    st.subheader("Backtesting con datos históricos de Coinbase")
    st.info("**Qué sí mide:** si el modelo (distancia al umbral + volatilidad + score de tendencia) estima mejor "
            "que un benchmark sin tendencia, con separación cronológica train/test y sin información futura.\n\n"
            "**Qué NO mide:** precios históricos reales de Kalshi (no hay histórico público gratuito). El P&L usa "
            "un precio de mercado SIMULADO (benchmark sin tendencia ± spread) más comisión y deslizamiento. "
            "Los resultados pasados no garantizan resultados futuros.")
    hours = st.slider("Horas de historial a descargar", 24, 336, 96, step=24)
    if st.button("Ejecutar backtest", type="primary"):
        try:
            with st.spinner("Descargando velas de Coinbase y simulando…"):
                dfh = fetch_history(hours)
                bt = run_backtest(dfh, cfg)
            st.session_state.bt = bt
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo completar el backtest: {e}")
            return
    bt = st.session_state.bt
    if not bt:
        return
    v = bt["validated"]
    st.caption(f"Periodo: {loc(bt['periodo'][0], '%Y-%m-%d %H:%M')} → {loc(bt['periodo'][1], '%Y-%m-%d %H:%M')} · "
               f"minutos rellenados por huecos: {bt['gaps']}")
    te, tr = bt["test"], bt["train"]
    st.markdown("#### Evaluación (datos no usados para ajustar)")
    a, b = st.columns(2)
    a.metric("Decisiones evaluadas", te["decisiones"])
    b.metric("Precisión direccional", f"{te['precision_direccional'] * 100:.1f}%",
             delta=f"{(te['precision_direccional'] - te['precision_persistencia']) * 100:+.1f} pts vs persistencia")
    c, d = st.columns(2)
    c.metric("Brier (usado)", f"{te['brier_usado']:.4f}")
    d.metric("Brier sin tendencia", f"{te['brier_sin_tendencia']:.4f}",
             help="Cuanto más bajo, mejor. 0.25 equivale a predecir siempre 50 %.")
    e, f_ = st.columns(2)
    e.metric("Señales generadas", te["señales"])
    f_.metric("Señales acertadas", te["aciertos"])
    lo, hi = te["ic95"]
    g, h = st.columns(2)
    g.metric("Tasa de acierto", "—" if te["señales"] == 0 else f"{te['tasa_acierto'] * 100:.1f}%",
             help="" if te["señales"] == 0 else f"IC 95 %: {lo * 100:.0f}%–{hi * 100:.0f}%")
    h.metric("P&L sintético medio", "—" if te["señales"] == 0 else f"{te['pnl_medio'] * 100:+.1f}¢")
    st.markdown(
        f"- **¿El score de tendencia aportó sobre el benchmark?** (decidido con TRAIN): "
        f"{'sí' if v['drift_supported'] else 'no → el modelo en vivo se usará SIN tendencia'}\n"
        f"- **¿Calibración de Platt útil?** {'sí' if v['platt_ok'] else 'no'}\n"
        f"- **Probabilidades validadas como calibradas:** {'sí' if v['calibrated'] else 'no'} "
        f"(exige ≥ 200 decisiones, Brier < 0.25 y no peor que el benchmark)")
    if te["señales"] < 30:
        st.warning("Hay pocas señales en el tramo de evaluación: la tasa de acierto no es estadísticamente fiable.")
    st.markdown("**Comparación train vs test**")
    keys = ["decisiones", "ventanas", "precision_direccional", "precision_persistencia", "brier_modelo",
            "brier_usado", "brier_sin_tendencia", "brier_constante_0.5", "señales", "aciertos", "tasa_acierto",
            "pnl_total", "pnl_medio"]
    st.dataframe(pd.DataFrame({"train": [tr[k] for k in keys], "test": [te[k] for k in keys]},
                              index=keys).astype(float).round(4))
    st.markdown("**Tabla de fiabilidad (test)** — la probabilidad media predicha debería parecerse a la frecuencia real")
    st.dataframe(bt["reliability"].round(3), hide_index=True)
    st.caption("Con muestras pequeñas el ruido es grande. Las decisiones de una misma ventana están correlacionadas.")


def render_help() -> None:
    st.markdown("""
### Cómo interpretar la señal
* **BUY UP** — El modelo da a YES más probabilidad que su precio de compra + comisión + deslizamiento, con al menos
  la ventaja mínima configurada y **todos** los filtros en verde.
* **BUY DOWN** — Lo mismo para el lado NO.
* **ESPERAR** — Algún filtro bloqueante falló. Abre **🔎 Diagnóstico** para ver cuál, con el valor leído y su límite.

### Una probabilidad alta NO es una señal
Si el modelo da 85 % y Kalshi vende ese lado a 86¢, comprarlo no tiene ventaja: pagas 86¢ + comisión + deslizamiento por
algo que vale 85¢. Para señal hace falta que la probabilidad supere el **costo de equilibrio + la ventaja exigida**
(la tabla de costos del diagnóstico muestra esa "probabilidad mínima"). Con precios de 80–90¢ y 6¢ de margen, la
probabilidad requerida suele rondar 88–98 %. Los filtros no se desactivan solos ni se relajan automáticamente.

### Qué hace cada filtro bloqueante
Datos frescos (precio, última operación, velas, Kalshi, reloj) · contrato correcto y reglas soportadas · referencia
coherente · precios YES/NO coherentes · tiempo restante · ventaja neta · rango de entrada · spread · incertidumbre.

### Qué pasaría si se relajaran (por eso no se tocan por defecto)
* **Ventaja mínima / margen por no calibrado:** bajarlos genera más señales, pero con ventajas que la comisión y el
  deslizamiento pueden borrar; en muestras pequeñas el modelo puede estar sobreconfiado.
* **Spread máximo:** subirlo te hace pagar más por entrar y salir.
* **Incertidumbre 'Alta':** se activa con varios factores a la vez (modelo sin validar, cerca del umbral, spread
  ancho, menos de 2 min, error Coinbase↔BRTI comparable al movimiento). Ignorarla es operar justo cuando el modelo
  es menos fiable.
* **Frescura de datos:** subir el límite deja operar con precios viejos.

### CAMBIOS de esta versión
1. **Todos los filtros se evalúan siempre**: antes el primero que fallaba ocultaba el resto del diagnóstico.
2. **La vela en formación ya no bloquea**: Coinbase la omite los primeros segundos de cada minuto y solo se usaba para dibujar.
3. **Reloj**: ahora se mide con la hora de descarga (antes se comparaba con respuestas en caché, lo que podía dar falsos desfases).
4. **Nuevos filtros de seguridad**: antigüedad de la última operación de Coinbase, referencia coherente con el precio
   (error de escala/contrato), coherencia de los 4 precios YES/NO, y contrato elegido = contrato analizado.
5. **Precios con décimas de centavo** (Kalshi cotiza 91.7¢ sobre 90¢): ya no se redondean a ciegas.
6. Zona horaria por defecto: America/Chicago. Gráfica estática y columnas lado a lado para celular.

### Limitaciones importantes
* Kalshi liquida con **CF Benchmarks BRTI** (promedio de ~60 s). Este analizador usa **Coinbase** como aproximación.
* La comisión usa la fórmula general taker `0.07·P·(1−P)` (configurable). Verifica la tabla vigente de Kalshi.
* Los precios YES/NO vienen de la API pública de mercados; la profundidad a ese precio puede ser pequeña.
* El score de tendencia/momentum es heurístico. Hasta que el backtest lo respalde, la probabilidad se marca como
  **no calibrada** y se exige un margen extra.
* Mercados de 15 minutos son muy eficientes y las comisiones consumen gran parte de cualquier ventaja pequeña.
  **No hay señales infalibles. Esto no es asesoría financiera. Puedes perder todo lo que operes.**
* Este programa **nunca envía órdenes**. No requiere ni guarda claves.
""")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Kalshi BTC 15m · Analizador", page_icon="🟣", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    st.markdown("### 🟣 Kalshi · BTC 15 min")
    st.caption("Solo análisis: no opera por ti. Ajustes: botón » arriba a la izquierda.")

    global LOCAL_TZ
    with st.sidebar:
        st.header("Configuración")
        tzname = st.selectbox("Zona horaria", TZ_CHOICES,
                              index=TZ_CHOICES.index(DEFAULT_TZ) if DEFAULT_TZ in TZ_CHOICES else 0)
        LOCAL_TZ = ZoneInfo(tzname)
        refresh = st.slider("Actualizar cada (s)", 5, 30, 8)
        forced = None
        try:
            mk = fetch_kalshi_markets()["markets"]
            tickers = [m["ticker"] for m in sorted(mk, key=lambda m: m["close_time"] or utcnow()) if m.get("ticker")]
            choice = st.selectbox("Contrato", ["Automático (el que está corriendo)"] + tickers)
            forced = None if choice.startswith("Automático") else choice
        except Exception:
            st.caption("No se pudo listar contratos de Kalshi ahora mismo.")
        with st.expander("Umbrales de la señal", expanded=False):
            min_edge = st.slider("Ventaja neta mínima (¢)", 1.0, 15.0, 4.0, 0.5)
            extra = st.slider("Margen extra si no está calibrado (¢)", 0.0, 10.0, 2.0, 0.5)
            max_spread = st.slider("Spread máximo (¢)", 1, 20, 8)
            slip = st.slider("Deslizamiento supuesto (¢)", 0.0, 5.0, 1.0, 0.5)
            fee_rate = st.number_input("Tasa de comisión taker", 0.0, 0.2, 0.07, 0.005, format="%.3f")
            min_tau = st.slider("Segundos mínimos restantes", 15, 180, 45)
        with st.expander("Frescura de datos", expanded=False):
            max_age = st.slider("Antigüedad máxima de precio/Kalshi (s)", 10, 60, 20)
            max_trade = st.slider("Antigüedad máxima de la última operación (s)", 20, 180, 60)
            st.caption("Subirlos permite señales con datos más viejos. Si tu conexión es lenta, súbelos poco a poco "
                       "y vigila la hora de actualización en el diagnóstico.")
        with st.expander("Modelo", expanded=False):
            basis = st.slider("Error Coinbase↔BRTI (puntos base)", 0.0, 6.0, 1.5, 0.5)
            drift = st.slider("Peso máximo del score (σ)", 0.0, 0.6, 0.25, 0.05)
            infl = st.slider("Inflación de volatilidad", 1.0, 1.5, 1.10, 0.05)
            use_bt = st.checkbox("Aplicar resultados del último backtest", True)
        chart_n = st.slider("Velas en la gráfica", 30, 240, 60, 10)
        st.divider()
        st.caption("Sin claves de API: se usan solo endpoints públicos. Nada se compra automáticamente.")

    cfg = Cfg(refresh_s=refresh, min_edge=min_edge / 100, uncal_extra=extra / 100, max_spread=max_spread / 100,
              fee_rate=fee_rate, slippage=slip / 100, min_tau_min=min_tau / 60, basis_bps=basis, drift_cap=drift,
              sigma_inflate=infl, chart_minutes=chart_n, use_backtest=use_bt, max_age_s=max_age,
              max_trade_age_s=max_trade)

    t_live, t_bt, t_help = st.tabs(["Señal", "Backtest", "Ayuda"])
    with t_live:
        live = st.fragment(run_every=cfg.refresh_s)(render_live)
        live(cfg, forced)
    with t_bt:
        render_backtest(cfg)
    with t_help:
        render_help()


if __name__ == "__main__":
    main()
