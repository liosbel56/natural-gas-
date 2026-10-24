# -*- coding: utf-8 -*-
"""
Analizador de señales — Kalshi BTC 15 minutos (serie KXBTC15M)
================================================================
* Es un ANALIZADOR. No compra ni vende nada. No necesita claves de API.
* Datos de precio: Coinbase Exchange (API pública).
* Datos del contrato: Kalshi (API pública de mercado, sin autenticación).
* Los resultados pasados NO garantizan resultados futuros. Esto no es asesoría financiera.

Ejecutar:   streamlit run app.py
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

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
LOCAL_TZ = ZoneInfo(os.getenv("APP_TZ", "America/New_York"))
TZ_CHOICES = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "America/Mexico_City",
              "America/Bogota", "America/Lima", "America/Santiago", "America/Argentina/Buenos_Aires",
              "America/Caracas", "Europe/Madrid", "UTC"]
UA = {"User-Agent": "kalshi-btc15-analyzer/1.0", "Accept": "application/json"}

GROUPS = [  # (etiqueta, columna, peso)  — 5 grupos, no 12 indicadores sueltos
    ("Tendencia (EMA 9/21/50 y pendiente de EMA 21)", "g_trend", 0.25),
    ("Momentum (histograma MACD y momentum 3/5 velas)", "g_mom", 0.30),
    ("Sobreextensión RSI(14) (contraria)", "g_rsi", 0.10),
    ("Volumen que confirma el movimiento", "g_vol", 0.15),
    ("Estructura (rupturas, soportes/resistencias)", "g_struct", 0.20),
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
    max_age_s: int = 20           # antigüedad máxima aceptada de precio/Kalshi
    chart_minutes: int = 90
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
    """Kalshi expone precios en dólares (*_dollars, texto) o en centavos (enteros). Acepta ambos."""
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
    yb, ya = _kprice(m, "yes_bid"), _kprice(m, "yes_ask")
    nb, na = _kprice(m, "no_bid"), _kprice(m, "no_ask")
    if ya is None and nb is not None:   # en un mercado binario: ask(YES) = 1 - bid(NO)
        ya = round(1 - nb, 4)
    if na is None and yb is not None:
        na = round(1 - yb, 4)
    if yb is None and na is not None:
        yb = round(1 - na, 4)
    if nb is None and ya is not None:
        nb = round(1 - ya, 4)
    rules = ((m.get("rules_primary") or "") + " " + (m.get("rules_secondary") or "")).strip()
    return {
        "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "title": m.get("title") or "",
        "subtitle": m.get("yes_sub_title") or "", "status": m.get("status"),
        "open_time": _ktime(m.get("open_time")), "close_time": _ktime(m.get("close_time")),
        "expiration_time": _ktime(m.get("expiration_time") or m.get("latest_expiration_time")),
        "strike": _to_float(m.get("floor_strike")), "strike_type": (m.get("strike_type") or "").lower() or None,
        "rules": rules, "yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na,
        "volume": _knum(m, ["volume_fp", "volume"]), "open_interest": _knum(m, ["open_interest_fp", "open_interest"]),
        "liquidity": (_knum(m, ["liquidity_dollars"]) if _knum(m, ["liquidity_dollars"]) is not None
                      else (None if _knum(m, ["liquidity"]) is None else _knum(m, ["liquidity"]) / 100.0)),
    }


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
# COSTES, INCERTIDUMBRE Y DECISIÓN
# ──────────────────────────────────────────────────────────────────────────────
def kalshi_fee(price: float, rate: float) -> float:
    """Comisión taker por contrato (sin redondeo): rate · P · (1 − P)."""
    return rate * price * (1 - price)


def kalshi_fee_order(contracts: int, price: float, rate: float) -> float:
    """Comisión de una orden: redondeada hacia arriba al siguiente centavo."""
    return math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100


def uncertainty(z: float, sd: float, sigma_ratio: float, spread: Optional[float],
                tau_min: float, calibrated: bool, cfg: Cfg) -> Tuple[str, List[str]]:
    pts, why = 0, []
    if not calibrated:
        pts += 1; why.append("probabilidad sin validar con backtest")
    if abs(z) < 0.5:
        pts += 1; why.append("el precio está muy cerca del umbral")
    if sigma_ratio is not None and np.isfinite(sigma_ratio) and sigma_ratio > 1.6:
        pts += 1; why.append("volatilidad reciente inestable")
    if spread is not None and spread > 0.05:
        pts += 1; why.append(f"spread amplio ({spread * 100:.0f}¢)")
    if tau_min < 2:
        pts += 1; why.append("quedan menos de 2 min: el error de referencia pesa más")
    if sd > 0 and (cfg.basis_bps / 1e4) / sd > 0.6:
        pts += 1; why.append("el error Coinbase↔BRTI es comparable al movimiento esperado")
    label = "Baja" if pts <= 1 else "Media" if pts <= 3 else "Alta"
    return label, why


def decide(p_yes: float, q: dict, tau_min: float, cfg: Cfg, calibrated: bool,
           problems: List[str], unc_label: str) -> dict:
    out = {"signal": "ESPERAR", "side": None, "reasons": [], "sides": {}, "req_edge": None}
    req = cfg.min_edge + (0.0 if calibrated else cfg.uncal_extra)
    out["req_edge"] = req
    for side, p, ask, bid in (("UP", p_yes, q.get("yes_ask"), q.get("yes_bid")),
                              ("DOWN", 1 - p_yes, q.get("no_ask"), q.get("no_bid"))):
        if ask is None:
            out["sides"][side] = None
            continue
        fee = kalshi_fee(ask, cfg.fee_rate)
        out["sides"][side] = {
            "p": p, "ask": ask, "bid": bid, "fee": fee,
            "spread": (ask - bid) if bid is not None else None,
            "breakeven": ask + fee + cfg.slippage,
            "edge": p - ask - fee - cfg.slippage,
            "win": 1 - ask - fee - cfg.slippage, "loss": -(ask + fee + cfg.slippage),
        }
    if problems:
        out["reasons"] = list(problems)
        return out
    valid = {k: v for k, v in out["sides"].items() if v}
    if not valid:
        out["reasons"] = ["Kalshi no muestra precios de compra válidos."]
        return out
    if tau_min < cfg.min_tau_min:
        out["reasons"] = [f"Quedan {tau_min * 60:.0f} s: demasiado cerca del vencimiento para emitir señal."]
        return out
    side, s = max(valid.items(), key=lambda kv: kv[1]["edge"])
    fails = []
    if s["edge"] < req:
        fails.append(f"Mejor opción ({side}) con ventaja neta {s['edge'] * 100:+.1f}¢; se exigen ≥ {req * 100:.1f}¢.")
    if not (cfg.min_entry <= s["ask"] <= cfg.max_entry):
        fails.append(f"Precio de entrada {s['ask'] * 100:.0f}¢ fuera del rango permitido.")
    if s["spread"] is not None and s["spread"] > cfg.max_spread:
        fails.append(f"Spread de {s['spread'] * 100:.0f}¢ supera el máximo de {cfg.max_spread * 100:.0f}¢.")
    if unc_label == "Alta":
        fails.append("Incertidumbre alta.")
    if fails:
        out["reasons"] = fails
        return out
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
        unc, _ = uncertainty(r["z"], r["sd"], r["sratio"], 2 * hs, r["tau"], True, cfg)
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
             "ventaja", "seg_restantes", "incertidumbre", "motivo", "cierre_utc", "resultado", "acierto"]


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


def cents(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.0f}¢"


CSS = """
<style>
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #2a1a52 0%, #0e0b1a 55%) fixed;}
section[data-testid="stSidebar"] {background:#120e24;border-right:1px solid #2b2150;}
h1,h2,h3 {color:#e9e5f5;}
div[data-testid="stMetric"] {background:#1a1530;border:1px solid #2f2459;border-radius:12px;padding:10px 14px;}
div[data-testid="stMetricValue"] {color:#c4b5fd;}
.stButton>button[kind="primary"] {background:#7c3aed;border:0;}
.sig {border:2px solid;border-radius:18px;padding:18px 22px;background:#161129;text-align:center;
      box-shadow:0 0 30px rgba(124,58,237,.25);}
.sig-label {color:#a78bfa;letter-spacing:.2em;font-size:.8rem;}
.sig-main {font-size:2.6rem;font-weight:800;line-height:1.1;}
.sig-sub {color:#cfc8ea;font-size:.95rem;margin-top:4px;}
.pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;margin-right:6px;}
.ok {background:#14532d;color:#bbf7d0;} .bad {background:#7f1d1d;color:#fecaca;} .old {background:#78350f;color:#fde68a;}
</style>
"""


def signal_card(signal: str, subtitle: str) -> str:
    color = {"BUY UP": "#22c55e", "BUY DOWN": "#ef4444", "ESPERAR": "#f59e0b"}.get(signal, "#a78bfa")
    return (f'<div class="sig" style="border-color:{color}"><div class="sig-label">SEÑAL</div>'
            f'<div class="sig-main" style="color:{color}">{signal}</div>'
            f'<div class="sig-sub">{subtitle}</div></div>')


def make_chart(f: pd.DataFrame, forming: Optional[pd.Series], ctr: Optional[Contract], minutes: int) -> go.Figure:
    d = f.tail(minutes).copy()
    tz = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    x = tz(d["time"])
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.50, 0.10, 0.14, 0.16, 0.10])
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
    fig.add_trace(go.Scatter(x=x, y=d["atr"], name="ATR 14", line=dict(color="#fbbf24")), row=5, col=1)
    if ctr is not None:
        for t0, label in ((ctr.window_start, "inicio"), (ctr.close_time, "cierre")):
            tx = pd.Timestamp(t0).tz_convert(LOCAL_TZ).tz_localize(None)
            if x.iloc[0] <= tx <= x.iloc[-1] + pd.Timedelta(minutes=16):
                fig.add_shape(type="line", x0=tx, x1=tx, y0=0, y1=1, xref="x", yref="paper",
                              line=dict(color="#a78bfa", dash="dot", width=1))
    fig.update_layout(template="plotly_dark", height=640, margin=dict(l=4, r=4, t=10, b=4), dragmode=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#120e24",
                      xaxis_rangeslider_visible=False, showlegend=False,
                      hovermode="x unified")
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
# VISTA EN VIVO
# ──────────────────────────────────────────────────────────────────────────────
def log_signal(rec: dict) -> None:
    hist: List[dict] = st.session_state.history
    last = next((h for h in reversed(hist) if h.get("contrato") == rec["contrato"]), None)
    now = utcnow()
    changed = last is None or last.get("señal") != rec["señal"]
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


def status_panel(now: datetime, cfg: Cfg) -> None:
    st.markdown("##### Estado de las conexiones")
    cols = st.columns(3)
    for col, name in zip(cols, ("Coinbase · precio", "Coinbase · velas 1m", "Kalshi · mercados")):
        s = st.session_state.status.get(name)
        with col:
            if not s:
                st.markdown(f"**{name}**<br><span class='pill old'>sin datos</span>", unsafe_allow_html=True)
                continue
            age = None if s["ok_at"] is None else (now - s["ok_at"]).total_seconds()
            if s["ok"]:
                pill = "<span class='pill ok'>OK</span>"
                extra = f"{s['latency_ms']:.0f} ms · actualizado {loc(s['ok_at'])}"
            else:
                pill = "<span class='pill bad'>ERROR</span>"
                extra = (f"último dato bueno ANTIGUO ({age:.0f} s): {loc(s['ok_at'])}" if age is not None
                         else "sin datos previos")
            st.markdown(f"**{name}** {pill}<br><small>{extra}</small>", unsafe_allow_html=True)
            if not s["ok"]:
                st.caption(s["error"])


def render_live(cfg: Cfg, forced: Optional[str]) -> None:
    now = utcnow()
    tick, tick_fail = safe_fetch("Coinbase · precio", fetch_ticker)
    cand, cand_fail = safe_fetch("Coinbase · velas 1m", fetch_candles)
    kal, kal_fail = safe_fetch("Kalshi · mercados", fetch_kalshi_markets)

    problems: List[str] = []
    age = lambda o: None if o is None else (now - o["fetched_at"]).total_seconds()

    # ── validar frescura ──
    if tick is None or tick_fail or age(tick) > cfg.max_age_s:
        problems.append("Precio de Coinbase no disponible o desactualizado.")
    if cand is None or cand_fail:
        problems.append("Velas de Coinbase no disponibles o desactualizadas.")
    if kal is None or kal_fail or age(kal) > cfg.max_age_s:
        problems.append("Datos de Kalshi no disponibles o desactualizados.")
    for o in (tick, cand, kal):
        sd = o.get("server_date") if o else None
        if sd is not None and abs((now - sd).total_seconds()) > 8:
            problems.append(f"El reloj de tu PC difiere ~{abs((now - sd).total_seconds()):.0f} s del servidor; sincronízalo.")
            break

    f = closed = forming = None
    row = None
    if cand is not None:
        df = cand["df"]
        end_closed = df["time"] + pd.Timedelta(seconds=60) <= pd.Timestamp(now)
        closed_raw, fm = df[end_closed], df[~end_closed]
        if len(closed_raw):
            closed, n_gap = fill_gaps(closed_raw.reset_index(drop=True))
            last_end = closed["time"].iloc[-1].to_pydatetime() + timedelta(seconds=60)
            if (now - last_end).total_seconds() > 150:
                problems.append("La última vela cerrada de Coinbase está atrasada (> 2.5 min).")
            f = compute_features(closed)
            row = f.iloc[-1]
            if len(f) < WARMUP or not bool(row["valid"]):
                problems.append(f"Datos insuficientes para los indicadores (se necesitan ≥ {WARMUP} velas cerradas).")
        if len(fm):
            forming = fm.iloc[-1]
        else:
            problems.append("No hay vela actual en la respuesta de Coinbase.")

    # ── contrato ──
    ctr: Optional[Contract] = None
    mk = None
    if kal is not None:
        mk = select_market(kal["markets"], now, forced)
        if mk is None:
            problems.append("No hay contratos KXBTC15M abiertos en este momento.")
    if mk is not None and closed is not None:
        ctr = build_contract(mk, closed, now)
        if not ctr.supported:
            problems.append("Reglas del contrato no soportadas por el analizador.")
        if ctr.reference is None:
            problems.append("No se pudo determinar el precio de referencia/umbral del contrato.")

    S0 = None
    if tick is not None:
        S0 = tick["price"]
    elif row is not None:
        S0 = float(row["close"])

    # ── modelo ──
    tau_min = None if ctr is None else (ctr.close_time - now).total_seconds() / 60.0
    mp, p_final, unc, why_unc, calibrated, notes_model = None, None, "Alta", [], False, []
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
        p_yes = mp["p"] if ctr.yes_if_above else 1 - mp["p"]
        if platt:
            p_yes = apply_platt(p_yes, *platt)
        p_final = p_yes
        q_tmp = {"yes_bid": mk["yes_bid"], "no_bid": mk["no_bid"], "yes_ask": mk["yes_ask"], "no_ask": mk["no_ask"]}
        spreads = [a - b for a, b in ((q_tmp["yes_ask"], q_tmp["yes_bid"]), (q_tmp["no_ask"], q_tmp["no_bid"]))
                   if a is not None and b is not None]
        unc, why_unc = uncertainty(mp["z"], mp["sd"], float(row["sigma_ratio"]), min(spreads) if spreads else None,
                                   tau_min, calibrated, cfg)

    q = {"yes_bid": None, "yes_ask": None, "no_bid": None, "no_ask": None}
    if mk is not None:
        q = {k: mk[k] for k in q}
    if p_final is None:
        dec = {"signal": "ESPERAR", "side": None, "sides": {}, "req_edge": None,
               "reasons": problems or ["Sin datos suficientes para estimar."]}
    else:
        dec = decide(p_final, q, tau_min, cfg, calibrated, problems, unc)
    signal = dec["signal"]

    # ── cabecera ──
    if True:
        if ctr is not None and tau_min is not None:
            sub = f"{ctr.ticker} · quedan {fmt_left(tau_min * 60)}"
        else:
            sub = "Sin contrato analizable"
        st.markdown(signal_card(signal, sub), unsafe_allow_html=True)
        st.caption(f"Última actualización: {loc(now)} (hora local) · se refresca cada {cfg.refresh_s} s")
    if True:
        c1, c2 = st.columns(2)
        c1.metric("BTC/USD (Coinbase)", "—" if S0 is None else f"${S0:,.2f}")
        c2.metric("Tiempo restante", "—" if tau_min is None else fmt_left(tau_min * 60))
        c3, c4 = st.columns(2)
        c3.metric("YES  compra / venta", f"{cents(q['yes_ask'])} / {cents(q['yes_bid'])}")
        c4.metric("NO  compra / venta", f"{cents(q['no_ask'])} / {cents(q['no_bid'])}")

    # ── explicación ──
    if p_final is not None and mp is not None:
        implied = None
        if q["yes_bid"] is not None and q["yes_ask"] is not None:
            implied = (q["yes_bid"] + q["yes_ask"]) / 2
        m1, m2 = st.columns(2)
        m1.metric("Prob. YES (modelo)", f"{p_final * 100:.1f}%", help="Probabilidad de que el contrato YES gane según el modelo.")
        m2.metric("Prob. implícita Kalshi", "—" if implied is None else f"{implied * 100:.1f}%")
        m3, m4 = st.columns(2)
        m3.metric("Incertidumbre", unc)
        m4.metric("Calibrada", "Sí" if calibrated else "No")
        st.markdown(f"**Dirección estimada:** {'ALCISTA (YES)' if p_final >= 0.5 else 'BAJISTA (NO)'}  ·  "
                    f"**Contrato:** {ctr.title} — referencia ${ctr.reference:,.2f} ({ctr.reference_kind})")
        # Dirección del PRECIO DE BTC a la que se refieren los factores:
        if dec["side"] is not None:      # señal activa: la del lado elegido
            dir_btc = 1 if (dec["side"] == "UP") == ctr.yes_if_above else -1
        else:                            # ESPERAR: la inclinación del modelo
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
        if why_unc:
            st.caption("Fuentes de incertidumbre: " + "; ".join(why_unc) + ".")
        for n_ in notes_model:
            st.caption(n_)
        # costes y beneficio por contrato
        shown = False
        for side, s_ in dec["sides"].items():
            if not s_:
                continue
            shown = True
            st.markdown(
                f"**{'BUY UP (YES)' if side == 'UP' else 'BUY DOWN (NO)'}** · modelo {s_['p'] * 100:.1f}% · "
                f"compra {cents(s_['ask'])} · spread {cents(s_['spread'])}  \n"
                f"comisión {s_['fee'] * 100:.2f}¢ · equilibrio {s_['breakeven'] * 100:.1f}% · "
                f"**ventaja neta {s_['edge'] * 100:+.1f}¢** · si gana {s_['win'] * 100:+.0f}¢ / si pierde {s_['loss'] * 100:+.0f}¢")
        if shown:
            ask_ref = next(s for s in dec["sides"].values() if s)["ask"]
            st.caption(f"Comisión de ejemplo para 10 contratos a {ask_ref * 100:.0f}¢: "
                       f"${kalshi_fee_order(10, ask_ref, cfg.fee_rate):.2f} (se redondea al centavo por orden). "
                       f"Incluye deslizamiento supuesto de {cfg.slippage * 100:.0f}¢.")
    else:
        st.markdown("**Motivo:** " + " ".join(dec["reasons"]))

    # ── gráfica ──
    if f is not None:
        st.caption("Velas 1 min · EMA 9 (rosa), 21 (morada), 50 (azul) · abajo: volumen, RSI, MACD, ATR")
        st.plotly_chart(make_chart(f, forming, ctr, cfg.chart_minutes),
                        config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False})
        last = f.iloc[-1]
        i1, i2, i3 = st.columns(3)
        i4, i5, i6 = st.columns(3)
        i1.metric("RSI 14", f"{last['rsi']:.1f}")
        i2.metric("MACD hist", f"{last['macd_hist']:.2f}")
        i3.metric("ATR 14", f"${last['atr']:.2f}")
        i4.metric("σ 1 min", f"{last['sigma'] * 1e4:.1f} pb")
        i5.metric("Vol. vs media", "—" if pd.isna(last["vol_ratio"]) else f"{last['vol_ratio']:.2f}×")
        i6.metric("Score", f"{last['score']:+.2f}")

    # ── contrato ──
    if mk is not None and ctr is not None:
        with st.expander("Datos y reglas del contrato de Kalshi"):
            st.write({"Ticker": ctr.ticker, "Título": ctr.title, "Inicio de ventana": loc(ctr.window_start),
                      "Cierre": loc(ctr.close_time), "Referencia": ctr.reference, "Tipo de referencia": ctr.reference_kind,
                      "YES gana si el precio al cierre es ≥ referencia": ctr.yes_if_above,
                      "Volumen": mk["volume"], "Interés abierto": mk["open_interest"], "Liquidez": mk["liquidity"]})
            for n_ in ctr.notes:
                st.caption("• " + n_)
            st.markdown("**Reglas oficiales (texto de Kalshi):**")
            st.write(ctr.rules or "(la API no devolvió texto de reglas)")

    status_panel(now, cfg)

    # ── historial ──
    if ctr is not None:
        log_signal({"hora_utc": now.strftime("%Y-%m-%d %H:%M:%S"), "contrato": ctr.ticker, "señal": signal,
                    "btc": None if S0 is None else round(S0, 2), "referencia": ctr.reference,
                    "yes_ask": q["yes_ask"], "no_ask": q["no_ask"],
                    "prob_modelo": None if p_final is None else round(p_final, 4),
                    "ventaja": None if not dec.get("side") else round(dec["sides"][dec["side"]]["edge"], 4),
                    "seg_restantes": None if tau_min is None else round(tau_min * 60),
                    "incertidumbre": unc, "motivo": " ".join(dec["reasons"])[:200],
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
    st.dataframe(h.iloc[::-1].head(200), hide_index=True)
    st.download_button("Descargar historial (CSV)", h.to_csv(index=False).encode("utf-8"),
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
    st.markdown("#### Resultados en el tramo de EVALUACIÓN (datos no usados para ajustar)")
    a, b, c, d = st.columns(4)
    a.metric("Decisiones evaluadas", te["decisiones"])
    b.metric("Precisión direccional", f"{te['precision_direccional'] * 100:.1f}%",
             delta=f"{(te['precision_direccional'] - te['precision_persistencia']) * 100:+.1f} pts vs persistencia")
    c.metric("Brier (usado)", f"{te['brier_usado']:.4f}")
    d.metric("Brier sin tendencia", f"{te['brier_sin_tendencia']:.4f}",
             help="Cuanto más bajo, mejor. 0.25 equivale a predecir siempre 50 %.")
    e, f_, g, h = st.columns(4)
    e.metric("Señales generadas", te["señales"])
    f_.metric("Señales acertadas", te["aciertos"])
    lo, hi = te["ic95"]
    g.metric("Tasa de acierto", "—" if te["señales"] == 0 else f"{te['tasa_acierto'] * 100:.1f}%",
             help="" if te["señales"] == 0 else f"IC 95 %: {lo * 100:.0f}%–{hi * 100:.0f}%")
    h.metric("P&L sintético medio / señal", "—" if te["señales"] == 0 else f"{te['pnl_medio'] * 100:+.1f}¢")
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
    st.dataframe(pd.DataFrame({"train (en muestra)": [tr[k] for k in keys], "test (fuera de muestra)": [te[k] for k in keys]},
                              index=keys).astype(float).round(4))
    st.markdown("**Tabla de fiabilidad (test)** — la probabilidad media predicha debería parecerse a la frecuencia real")
    st.dataframe(bt["reliability"].round(3), hide_index=True)
    st.caption("Con muestras pequeñas el ruido es grande. Las decisiones de una misma ventana están correlacionadas.")
    st.session_state["bt_ready"] = True


def render_help() -> None:
    st.markdown("""
### Cómo interpretar la señal
* **BUY UP** — La comprobación de YES (el precio al cierre ≥ referencia) tiene, según el modelo, más probabilidad que el
  precio de compra de YES + comisión + deslizamiento, con al menos la ventaja mínima configurada.
* **BUY DOWN** — Lo mismo para el lado NO.
* **ESPERAR** — No hay ventaja suficiente, los datos no son fiables/están atrasados, el spread es amplio, quedan menos
  de 45 s o la incertidumbre es alta. **Es el resultado más frecuente y es lo esperado.**

### Limitaciones importantes
* Kalshi liquida con **CF Benchmarks BRTI** (promedio de ~60 s). Este analizador usa **Coinbase** como aproximación; la
  diferencia (la "base") se modela con un margen configurable, pero puede importar cuando el precio está cerca del umbral.
* La comisión usa la fórmula general taker de Kalshi `0.07·P·(1−P)` (configurable). Kalshi puede cambiar su tarifa o aplicar
  otra a una serie concreta: verifica su tabla de comisiones vigente.
* Los precios YES/NO provienen de la API pública de mercados de Kalshi; no se consulta el libro de órdenes completo, por lo
  que la profundidad disponible a ese precio puede ser pequeña.
* El score de tendencia/momentum es heurístico (pesos fijos, no ajustados). Hasta que el backtest lo respalde, la
  probabilidad se marca como **no calibrada** y se exige un margen extra.
* Mercados de 15 minutos son muy eficientes y las comisiones consumen gran parte de cualquier ventaja pequeña.
  **No hay señales infalibles. Esto no es asesoría financiera. Puedes perder todo lo que operes.**
* Este programa **nunca envía órdenes**. No requiere ni guarda claves.
""")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Kalshi BTC 15m · Analizador", page_icon="🟣", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    st.markdown("### 🟣 Kalshi · BTC 15 min")
    st.caption("Solo análisis: no opera por ti. Ajustes: botón » arriba a la izquierda.")

    global LOCAL_TZ
    with st.sidebar:
        st.header("Configuración")
        tzname = st.selectbox("Zona horaria", TZ_CHOICES, index=TZ_CHOICES.index(os.getenv("APP_TZ", "America/New_York"))
                              if os.getenv("APP_TZ", "America/New_York") in TZ_CHOICES else 0)
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
              sigma_inflate=infl, chart_minutes=chart_n, use_backtest=use_bt)

    t_live, t_bt, t_help = st.tabs(["Señal en vivo", "Backtesting", "Ayuda y límites"])
    with t_live:
        live = st.fragment(run_every=cfg.refresh_s)(render_live)
        live(cfg, forced)
    with t_bt:
        render_backtest(cfg)
    with t_help:
        render_help()


if __name__ == "__main__":
    main()
