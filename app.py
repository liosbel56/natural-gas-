# -*- coding: utf-8 -*-
"""
Kalshi Signal Bot — SOLO BTC/USD · contratos KXBTC15M (15 minutos)
===================================================================
* ANÁLISIS: solo con endpoints públicos, sin claves.
* OPERACIÓN (opcional, APAGADA por defecto): puede enviar órdenes a Kalshi con TUS claves (variables de entorno).
  Modos: Apagado · Simulación (no envía nada) · REAL (requiere KALSHI_LIVE_OK=1 y confirmación escrita).
* Precio/velas: Coinbase (Exchange API; respaldo: API pública de Advanced Trade). Contrato: Kalshi (público).
* Los resultados pasados NO garantizan resultados futuros. Esto no es asesoría financiera.

NOVEDADES de esta versión:
  - Muestra la INCLINACIÓN (UP/DOWN) y cuántos ¢ de ventaja FALTAN cuando la señal es ESPERAR.
  - Perfiles de sensibilidad (Conservador / Equilibrado / Sensible) en la barra lateral.
  - SEÑAL TEMPRANA: una dirección (UP/DOWN) por cada ventana de 15 min, emitida al inicio, sin exigir 80–90 %.

Instalar:        pip install "streamlit>=1.37" pandas numpy requests plotly cryptography
Ejecutar:        streamlit run app.py
Auto-pruebas:    python app.py --selftest        (datos SINTÉTICOS, no necesita internet)
Diagnóstico red: python app.py --live-check      (consulta Coinbase y Kalshi UNA vez e imprime el retraso real)
"""
from __future__ import annotations

import base64
import html as _html
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
COINBASE_BASE = "https://api.exchange.coinbase.com"
COINBASE_ADV_BASE = "https://api.coinbase.com"          # respaldo: endpoints públicos de mercado (sin clave)
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
# "no-cache" pide a intermediarios (CDN) no servir copias viejas; no todos lo respetan, por eso se mide la edad real.
UA = {"User-Agent": "kalshi-btc15-analyzer/2.1", "Accept": "application/json",
      "Cache-Control": "no-cache", "Pragma": "no-cache"}
PRICE_KEYS = ("yes_bid", "yes_ask", "no_bid", "no_ask")
CANDLE_SPAN_MIN = 290         # minutos pedidos a Coinbase (límite de la API: 300 velas por solicitud)
HTTP_TIMEOUT = (3.5, 7.0)     # (conexión, lectura) en segundos
HTTP_RETRIES = 2              # reintentos controlados (con espera creciente)

GROUPS = [  # (etiqueta, columna, peso)  — 5 grupos, no 12 indicadores sueltos
    ("Tendencia (EMA 9/21/50 y pendiente de EMA 21)", "g_trend", 0.25),
    ("Momentum (histograma MACD y momentum 3/5 velas)", "g_mom", 0.30),
    ("Sobreextensión RSI(14) (contraria)", "g_rsi", 0.10),
    ("Volumen que confirma el movimiento", "g_vol", 0.15),
    ("Estructura (rupturas, soportes/resistencias)", "g_struct", 0.20),
]

DECISION_FILTERS = [
    ("quotes", "Precios de compra de Kalshi"),
    ("time_left", "Tiempo restante"),
    ("edge", "Ventaja neta mínima"),
    ("entry", "Rango del precio de entrada"),
    ("spread", "Spread máximo"),
    ("depth", "Liquidez (contratos disponibles)"),
    ("uncertainty", "Incertidumbre"),
]

# Perfiles de sensibilidad: (ventaja mínima ¢, margen extra si no calibrado ¢, deslizamiento ¢)
PROFILES = {"Conservador": (4.0, 2.0, 1.0), "Equilibrado": (2.5, 1.0, 0.5), "Sensible": (1.5, 0.0, 0.5)}


@dataclass
class Cfg:
    refresh_s: int = 6
    min_edge: float = 0.04        # ventaja mínima neta (en $ por contrato)
    uncal_extra: float = 0.02     # margen extra mientras el modelo no esté validado
    max_spread: float = 0.08
    min_entry: float = 0.05
    max_entry: float = 0.95
    min_depth: float = 5.0        # contratos mínimos disponibles al precio de entrada (si Kalshi informa el tamaño)
    fee_rate: float = 0.07        # comisión taker general de Kalshi: rate*P*(1-P)
    slippage: float = 0.01
    min_tau_min: float = 0.75     # no emitir señal con menos de 45 s
    basis_bps: float = 1.5        # incertidumbre Coinbase vs BRTI (puntos base)
    drift_cap: float = 0.25       # máximo desplazamiento (en desv. típicas) que puede aportar el score
    sigma_inflate: float = 1.10   # colas gruesas
    max_age_s: int = 20           # antigüedad máxima del precio / datos de Kalshi (desde la descarga)
    max_trade_age_s: int = 60     # antigüedad máxima de la última operación de Coinbase
    candle_warn_s: int = 75       # vela cerrada: amarillo si su antigüedad supera esto
    candle_max_s: int = 150       # vela cerrada: ROJO y bloqueo si su antigüedad supera esto
    stale_trigger_s: int = 90     # si la fuente principal supera esto, se consulta también la de respaldo
    max_clock_skew_s: float = 8.0
    max_ref_dev: float = 0.03
    quote_tol: float = 0.02
    chart_minutes: int = 60
    use_backtest: bool = True
    sim_half_spread: float = 0.01
    early_on: bool = True          # emitir señal temprana (1 por contrato de 15 min)
    early_after_s: int = 30        # segundos tras el inicio de la ventana para emitirla
    early_lock: bool = True        # mantener la dirección durante toda la ventana
    early_drift: float = 0.60      # peso de la tendencia (σ) en la señal temprana


# ──────────────────────────────────────────────────────────────────────────────
# HTTP (sin dependencias de Streamlit: se puede usar desde --live-check)
# ──────────────────────────────────────────────────────────────────────────────
_SESSION: Optional[requests.Session] = None
_BACKOFF: Dict[str, float] = {}   # base_url -> instante (epoch) hasta el que NO se consulta tras un HTTP 429


def get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
        _SESSION = s
    return _SESSION


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def http_get(base: str, path: str, params: Optional[dict] = None, timeout: Tuple[float, float] = HTTP_TIMEOUT,
             retries: int = HTTP_RETRIES) -> Tuple[Any, float, Optional[datetime], dict]:
    """GET con reintentos controlados (solo errores de red y 5xx), respeto de HTTP 429 y metadatos de caché.

    Devuelve (json, latencia_ms, cabecera Date del servidor, meta). meta incluye 'age' (cabecera Age, segundos
    que la respuesta estuvo en caché de un intermediario) y 'cache' (estado de caché si el servidor lo informa)."""
    wait = _BACKOFF.get(base, 0.0) - time.time()
    if wait > 0:
        raise RuntimeError(f"En espera por límite de solicitudes (HTTP 429): {wait:.0f} s más antes de reintentar.")
    last_err = "sin detalle"
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = get_session().get(base + path, params=params, headers=UA, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"{type(e).__name__}"
        else:
            lat = (time.time() - t0) * 1000
            if r.status_code == 429:
                ra = _to_float(r.headers.get("Retry-After"))
                _BACKOFF[base] = time.time() + min(max(ra if ra is not None else 15.0, 5.0), 120.0)
                raise RuntimeError("Límite de solicitudes alcanzado (HTTP 429). Se pausan las consultas a ese servidor "
                                   "unos segundos; aumenta el intervalo de actualización si se repite.")
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
            elif r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            else:
                try:
                    j = r.json()
                except ValueError:
                    raise RuntimeError("Respuesta incompleta o inválida (no es JSON).")
                server_dt = None
                try:
                    server_dt = parsedate_to_datetime(r.headers["Date"])
                except Exception:
                    pass
                meta = {"date": server_dt, "age": _to_float(r.headers.get("Age")),
                        "cache": r.headers.get("CF-Cache-Status") or r.headers.get("X-Cache"),
                        "attempts": attempt + 1}
                return j, lat, server_dt, meta
        if attempt < retries:
            time.sleep(0.3 * (2 ** attempt))
    raise RuntimeError(f"Sin respuesta válida tras {retries + 1} intentos ({last_err}).")


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ──────────────────────────────────────────────────────────────────────────────
# COINBASE: precio
# ──────────────────────────────────────────────────────────────────────────────
def fetch_ticker() -> dict:
    j, lat, sdt, meta = http_get(COINBASE_BASE, f"/products/{PRODUCT}/ticker")
    t_recv = utcnow()
    price = _to_float(j.get("price")) if isinstance(j, dict) else None
    if price is None or price <= 0:
        raise RuntimeError("Coinbase devolvió un precio inválido.")
    ex_time = None
    try:
        ex_time = pd.to_datetime(j.get("time"), utc=True).to_pydatetime()
    except Exception:
        pass
    return {"price": price, "bid": _to_float(j.get("bid")), "ask": _to_float(j.get("ask")),
            "exchange_time": ex_time, "fetched_at": t_recv, "latency_ms": lat, "server_date": sdt, "meta": meta}


# ──────────────────────────────────────────────────────────────────────────────
# COINBASE: velas (validación estricta de timestamps y de velas cerradas)
# ──────────────────────────────────────────────────────────────────────────────
def parse_candles(raw: Any, now: Optional[datetime] = None, max_future_s: float = 5.0) -> Tuple[pd.DataFrame, dict]:
    """[[time, low, high, open, close, volume], ...] con time = inicio de la vela en epoch UTC (segundos).

    Valida y CUENTA (no oculta): filas inválidas, desalineadas a minuto, futuras, duplicadas y desordenadas.
    No inventa ninguna vela."""
    now = now or utcnow()
    if not isinstance(raw, list) or len(raw) == 0:
        raise RuntimeError("Coinbase devolvió velas vacías o en formato inesperado.")
    try:
        arr = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        raise RuntimeError("Velas con valores no numéricos.")
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise RuntimeError("Las velas no tienen las 6 columnas esperadas.")
    d_ts = np.diff(arr[:, 0])
    rep = {"filas": int(len(arr)), "invalidas": 0, "desalineadas": 0, "futuras": 0, "duplicadas": 0,
           "desordenadas": bool(len(d_ts) and not (np.all(d_ts <= 0) or np.all(d_ts >= 0)))}
    df = pd.DataFrame(arr[:, :6], columns=["ts", "low", "high", "open", "close", "volume"])
    n0 = len(df)
    df = df[np.isfinite(df).all(axis=1)]
    df = df[(df[["low", "high", "open", "close"]] > 0).all(axis=1)]
    df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9)
            & (df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9)
            & (df["volume"] >= 0)]
    rep["invalidas"] = n0 - len(df)
    aligned = (df["ts"] % 60) == 0
    rep["desalineadas"] = int((~aligned).sum())
    df = df[aligned]
    fut = df["ts"] > now.timestamp() + max_future_s
    rep["futuras"] = int(fut.sum())
    df = df[~fut]
    dup = df.duplicated("ts", keep="last")
    rep["duplicadas"] = int(dup.sum())
    df = df[~dup]
    if df.empty:
        raise RuntimeError("Todas las velas recibidas fueron inválidas.")
    df = df.copy()
    df["time"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    df = df.drop(columns="ts").sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "volume"]], rep


def split_closed(df: pd.DataFrame, now: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa velas CERRADAS (inicio + 60 s ≤ ahora) de la que aún se forma. La hora de cierre de la vela
    NO es la hora de cierre del contrato."""
    ended = df["time"] + pd.Timedelta(seconds=60) <= pd.Timestamp(now)
    return df[ended].reset_index(drop=True), df[~ended].reset_index(drop=True)


def candle_age_s(closed: Optional[pd.DataFrame], now: datetime) -> float:
    """Segundos transcurridos desde el CIERRE de la última vela cerrada (inf si no hay)."""
    if closed is None or len(closed) == 0:
        return float("inf")
    end = closed["time"].iloc[-1].to_pydatetime() + timedelta(seconds=60)
    return (now - end).total_seconds()


def gap_stats(closed_raw: pd.DataFrame) -> Tuple[int, int]:
    """(minutos faltantes en todo el tramo, minutos faltantes en los últimos 10 intervalos). Coinbase omite
    minutos sin operaciones; aquí solo se cuentan, no se rellenan."""
    if len(closed_raw) < 2:
        return 0, 0
    steps = (closed_raw["time"].diff().dt.total_seconds().iloc[1:] / 60 - 1).clip(lower=0).round().astype(int)
    return int(steps.sum()), int(steps.tail(10).sum())


def fill_gaps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Rellena SOLO huecos INTERNOS (minutos sin operaciones: vela plana, volumen 0). Nunca extiende la serie
    hacia el presente: un retraso al final se ve como retraso."""
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


def _exchange_raw(now: datetime) -> Tuple[Any, float, Optional[datetime], dict]:
    # start/end EXPLÍCITOS: la consulta sin parámetros devuelve una respuesta compartida que puede venir de caché
    # y quedar atrás; con 'end' = ahora la URL cambia en cada solicitud y no se reutiliza una copia vieja.
    end = now.replace(microsecond=0)
    start = end - timedelta(minutes=CANDLE_SPAN_MIN)
    return http_get(COINBASE_BASE, f"/products/{PRODUCT}/candles",
                    params={"granularity": 60, "start": _iso(start), "end": _iso(end)})


def _advanced_to_raw(j: Any) -> List[List[float]]:
    c = j.get("candles") if isinstance(j, dict) else None
    if not isinstance(c, list) or not c:
        raise RuntimeError("Advanced Trade devolvió velas vacías o en formato inesperado.")
    out = []
    for x in c:
        try:
            out.append([float(x["start"]), float(x["low"]), float(x["high"]), float(x["open"]),
                        float(x["close"]), float(x["volume"])])
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise RuntimeError("Advanced Trade: ninguna vela con formato válido.")
    return out


def _advanced_raw(now: datetime) -> Tuple[Any, float, Optional[datetime], dict]:
    end = int(now.timestamp())
    start = end - CANDLE_SPAN_MIN * 60
    j, lat, sdt, meta = http_get(COINBASE_ADV_BASE, f"/api/v3/brokerage/market/products/{PRODUCT}/candles",
                                 params={"start": str(start), "end": str(end), "granularity": "ONE_MINUTE"})
    return _advanced_to_raw(j), lat, sdt, meta


CANDLE_SOURCES: List[Tuple[str, Callable[[datetime], Tuple[Any, float, Optional[datetime], dict]]]] = [
    ("Coinbase Exchange", _exchange_raw),
    ("Coinbase Advanced (respaldo)", _advanced_raw),
]


def fetch_candles(stale_trigger_s: float = 90.0) -> dict:
    """Descarga velas de 1 min. Consulta la fuente principal; si su última vela cerrada está atrasada (o falla),
    consulta también la de respaldo y usa la MÁS RECIENTE. Nunca mezcla fuentes ni fabrica velas.
    Cada intento queda registrado en 'attempts' para el diagnóstico."""
    attempts: List[dict] = []
    best: Optional[dict] = None
    for name, fn in CANDLE_SOURCES:
        t_req = utcnow()
        try:
            j, lat, sdt, meta = fn(t_req)
            t_recv = utcnow()
            df, rep = parse_candles(j, t_recv)
            closed, _ = split_closed(df, t_recv)
            age = candle_age_s(closed, t_recv)
            resp_age = None if sdt is None else (t_recv - sdt).total_seconds()
            attempts.append({"source": name, "ok": True, "candle_age_s": age, "latency_ms": lat,
                             "resp_age_s": resp_age, "age_header": meta.get("age"), "cache": meta.get("cache"),
                             "report": rep, "error": None})
            cand = {"df": df, "fetched_at": t_recv, "latency_ms": lat, "server_date": sdt, "source": name,
                    "report": rep, "meta": meta}
            if best is None or age < best["_age"]:
                best = dict(cand, _age=age)
            if age <= stale_trigger_s:
                break           # suficientemente reciente: no hace falta consultar el respaldo
        except Exception as e:  # noqa: BLE001
            attempts.append({"source": name, "ok": False, "candle_age_s": None, "latency_ms": None,
                             "resp_age_s": None, "age_header": None, "cache": None, "report": None,
                             "error": f"{type(e).__name__}: {str(e)[:140]}"})
    if best is None:
        raise RuntimeError("Ninguna fuente de velas respondió: "
                           + " | ".join(f"{a['source']}: {a['error']}" for a in attempts))
    best.pop("_age", None)
    best["attempts"] = attempts
    return best


def fetch_history(hours: int) -> pd.DataFrame:
    """Historial de velas de 1 min paginando (máx. ~290 velas por solicitud). Solo velas cerradas."""
    now = utcnow()
    end = now.replace(second=0, microsecond=0)
    remaining = int(hours) * 60
    frames = []
    while remaining > 0:
        n = min(290, remaining)
        start = end - timedelta(minutes=n)
        j, _, _, _ = http_get(COINBASE_BASE, f"/products/{PRODUCT}/candles",
                              params={"granularity": 60, "start": _iso(start), "end": _iso(end)}, timeout=(3.5, 15))
        if isinstance(j, list) and j:
            frames.append(parse_candles(j, now)[0])
        end = start
        remaining -= n
        time.sleep(0.2)  # cortesía con el límite de la API pública
    if not frames:
        raise RuntimeError("No se pudo descargar historial de Coinbase.")
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return split_closed(df, utcnow())[0]


# ──────────────────────────────────────────────────────────────────────────────
# KALSHI
# ──────────────────────────────────────────────────────────────────────────────
def fetch_kalshi_markets() -> dict:
    j, lat, sdt, meta = http_get(KALSHI_BASE, "/markets",
                                 params={"series_ticker": SERIES, "status": "open", "limit": 50})
    t_recv = utcnow()
    markets = j.get("markets") if isinstance(j, dict) else None
    if not isinstance(markets, list):
        raise RuntimeError("Kalshi no devolvió la lista 'markets'.")
    return {"markets": [parse_market(m) for m in markets], "fetched_at": t_recv,
            "latency_ms": lat, "server_date": sdt, "meta": meta}


def fetch_kalshi_market_result(ticker: str) -> Optional[str]:
    j, _, _, _ = http_get(KALSHI_BASE, f"/markets/{ticker}")
    m = j.get("market", j) if isinstance(j, dict) else {}
    res = str(m.get("result") or "").lower()
    return res if res in ("yes", "no") else None


def _kprice(m: dict, key: str) -> Optional[float]:
    """Kalshi expone precios en dólares (*_dollars, texto). Se mantiene el respaldo en centavos.
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


def turn_alerts(f: pd.DataFrame) -> List[Tuple[str, int]]:
    """Señales de posible CAMBIO de dirección (solo informativas y para sumar incertidumbre; nunca generan una
    señal por sí solas). Devuelve [(texto, dirección_sugerida_para_BTC: +1 / -1 / 0)]."""
    out: List[Tuple[str, int]] = []
    if f is None or len(f) < 6:
        return out
    s = np.sign(f["ema9"] - f["ema21"]).tail(4).to_numpy()
    if s[-1] != 0 and s[-1] != s[0]:
        out.append(("EMA 9 cruzó la EMA 21 en las últimas 3 velas", int(s[-1])))
    h = f["macd_hist"].tail(4).to_numpy()
    if np.sign(h[-1]) != 0 and np.sign(h[-1]) != np.sign(h[0]):
        out.append(("El histograma MACD cambió de signo en las últimas 3 velas", int(np.sign(h[-1]))))
    ah = np.abs(f["macd_hist"].tail(4).to_numpy())
    sh = np.sign(f["macd_hist"].iloc[-1])
    if sh != 0 and np.all(np.diff(ah) < 0) and np.sign(f["macd_hist"].iloc[-4]) == sh:
        out.append(("El histograma MACD se debilita 3 velas seguidas (momentum agotándose)", int(-sh)))
    r0, r1 = float(f["rsi"].iloc[-2]), float(f["rsi"].iloc[-1])
    if r0 > 70 >= r1:
        out.append(("RSI sale de sobrecompra (posible giro a la baja)", -1))
    if r0 < 30 <= r1:
        out.append(("RSI sale de sobreventa (posible giro al alza)", 1))
    sr = f["sigma_ratio"].iloc[-1]
    if np.isfinite(sr) and sr > 1.6:
        out.append((f"La volatilidad reciente se disparó (σ reciente/σ 60 min = {sr:.2f})", 0))
    return out


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
    """Kalshi cotiza con décimas de centavo (p. ej. 91.7¢): no redondear a ciegas."""
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
# FRESCURA (semáforo verde / amarillo / rojo)
# ──────────────────────────────────────────────────────────────────────────────
def freshness(price_age: Optional[float], trade_age: Optional[float], candle_age: Optional[float],
              kalshi_age: Optional[float], conn_errors: List[str], cfg: Cfg) -> Tuple[str, List[str]]:
    """Devuelve ('green'|'yellow'|'red', motivos). Rojo = alguna fuente crítica está obsoleta o no responde."""
    rank = {"green": 0, "yellow": 1, "red": 2}
    level, why = "green", []

    def bump(lv: str, msg: str) -> None:
        nonlocal level
        if rank[lv] > rank[level]:
            level = lv
        why.append(msg)

    def chk(name: str, age: Optional[float], warn: float, bad: float) -> None:
        if age is None or not np.isfinite(age):
            bump("red", f"{name}: sin datos")
        elif age > bad:
            bump("red", f"{name}: {age:.0f} s (límite {bad:.0f} s)")
        elif age > warn:
            bump("yellow", f"{name}: {age:.0f} s (aviso desde {warn:.0f} s)")

    chk("Precio", price_age, cfg.max_age_s / 2, cfg.max_age_s)
    if trade_age is not None:
        chk("Última operación", trade_age, cfg.max_trade_age_s / 2, cfg.max_trade_age_s)
    chk("Vela cerrada", candle_age, cfg.candle_warn_s, cfg.candle_max_s)
    chk("Kalshi", kalshi_age, cfg.max_age_s / 2, cfg.max_age_s)
    for e in conn_errors:
        bump("yellow", f"Último intento fallido: {e}")
    return level, why


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
                tau_min: float, calibrated: bool, cfg: Cfg, turn_against: int = 0) -> Tuple[str, List[str], int]:
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
    if turn_against >= 1:
        pts += 1; why.append(f"{turn_against} señal(es) de posible cambio de dirección contraria(s) a la apuesta (+1)")
    label = "Baja" if pts <= 1 else "Media" if pts <= 3 else "Alta"
    return label, why, pts


def decide(p_yes: float, q: dict, tau_min: float, cfg: Cfg, calibrated: bool,
           problems: List[str], unc_label: str, unc_detail: str = "") -> dict:
    """Decisión final. Evalúa TODOS los filtros (no se detiene en el primero que falla).
    Solo hay señal si no hay problemas de datos y ningún filtro bloqueante falla."""
    req = cfg.min_edge + (0.0 if calibrated else cfg.uncal_extra)
    out: Dict[str, Any] = {"signal": "ESPERAR", "side": None, "best": None, "reasons": [], "sides": {},
                           "req_edge": req, "filters": []}
    for side, p, ask, bid, depth in (("UP", p_yes, q.get("yes_ask"), q.get("yes_bid"), q.get("yes_ask_size")),
                                     ("DOWN", 1 - p_yes, q.get("no_ask"), q.get("no_bid"), q.get("yes_bid_size"))):
        if ask is None:
            out["sides"][side] = None
            continue
        fee = kalshi_fee(ask, cfg.fee_rate)
        be = ask + fee + cfg.slippage
        out["sides"][side] = {
            "p": p, "ask": ask, "bid": bid, "fee": fee, "depth": depth,
            "spread": (ask - bid) if bid is not None else None,
            "breakeven": be, "edge": p - be, "req_p": be + req,
            "win": 1 - be, "loss": -be,
        }
    fl: List[dict] = out["filters"]
    valid = {k: v for k, v in out["sides"].items() if v}

    if valid:
        fl.append(mkf("quotes", "Precios de compra de Kalshi", "ok",
                      " · ".join(f"{k} {cents(v['ask'])}" for k, v in valid.items())))
    else:
        fl.append(mkf("quotes", "Precios de compra de Kalshi", "fail", "sin precios", "",
                      "Kalshi no muestra precios de compra válidos."))
    t_ok = tau_min >= cfg.min_tau_min
    fl.append(mkf("time_left", "Tiempo restante", "ok" if t_ok else "fail", f"{tau_min * 60:.0f} s",
                  f"≥ {cfg.min_tau_min * 60:.0f} s",
                  "" if t_ok else f"Quedan {tau_min * 60:.0f} s: demasiado cerca del vencimiento para emitir señal."))
    side = None
    if valid:
        side, s = max(valid.items(), key=lambda kv: kv[1]["edge"])
        out["best"] = side
        e_ok = s["edge"] >= req
        imposible = s["req_p"] > 0.995
        det = (f"Mejor opción ({side}): el modelo da {pct(s['p'])} y el costo de equilibrio es {pct(s['breakeven'])} "
               f"(compra {cents(s['ask'])} + comisión {s['fee'] * 100:.2f}¢ + deslizamiento {cents(cfg.slippage)}). "
               f"Ventaja neta {s['edge'] * 100:+.1f}¢; se exigen ≥ {req * 100:.1f}¢, o sea una probabilidad del modelo "
               f"≥ {pct(s['req_p'])}"
               + (" (el modelo nunca pasa de 99.5 %: a este precio no puede haber señal)." if imposible else "."))
        fl.append(mkf("edge", f"Ventaja neta mínima ({side})", "ok" if e_ok else "fail",
                      f"{s['edge'] * 100:+.1f}¢", f"≥ {req * 100:.1f}¢", "" if e_ok else det))
        r_ok = cfg.min_entry <= s["ask"] <= cfg.max_entry
        fl.append(mkf("entry", "Rango del precio de entrada", "ok" if r_ok else "fail", cents(s["ask"]),
                      f"{cfg.min_entry * 100:.0f}¢–{cfg.max_entry * 100:.0f}¢",
                      "" if r_ok else f"Precio de entrada {cents(s['ask'])} fuera del rango permitido."))
        if s["spread"] is None:
            fl.append(mkf("spread", "Spread máximo", "warn", "sin precio de venta", f"≤ {cfg.max_spread * 100:.0f}¢",
                          "No hay precio de venta para medir el spread (no bloquea).", blocking=False))
        else:
            sp_ok = s["spread"] <= cfg.max_spread
            fl.append(mkf("spread", "Spread máximo", "ok" if sp_ok else "fail", cents(s["spread"]),
                          f"≤ {cfg.max_spread * 100:.0f}¢",
                          "" if sp_ok else f"Spread de {cents(s['spread'])} supera el máximo de {cfg.max_spread * 100:.0f}¢."))
        if s["depth"] is None:
            fl.append(mkf("depth", "Liquidez (contratos disponibles)", "warn", "no informada", f"≥ {cfg.min_depth:.0f}",
                          "Kalshi no informó cuántos contratos hay a ese precio (no bloquea).", blocking=False))
        else:
            dp_ok = s["depth"] >= cfg.min_depth
            fl.append(mkf("depth", "Liquidez (contratos disponibles)", "ok" if dp_ok else "fail",
                          f"{s['depth']:,.0f} contratos", f"≥ {cfg.min_depth:.0f}",
                          "" if dp_ok else f"Solo hay {s['depth']:,.0f} contratos al precio de entrada: la orden "
                          "probablemente se ejecutaría a un precio peor."))
    else:
        for fid, name in DECISION_FILTERS[2:6]:
            fl.append(mkf(fid, name, "na", blocking=False))
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


def compute_lean(p_final: Optional[float], dec: dict) -> Optional[dict]:
    """INCLINACIÓN: hacia qué lado apunta el modelo cuando la señal es ESPERAR y cuántos ¢ de ventaja faltan.
    Es solo informativa: NO es una señal de compra y nunca dispara órdenes."""
    if p_final is None or dec.get("signal") != "ESPERAR" or dec.get("req_edge") is None:
        return None
    side = "UP" if p_final >= 0.5 else "DOWN"
    s = (dec.get("sides") or {}).get(side)
    if not s:
        return None
    return {"side": side, "p": s["p"], "edge": s["edge"], "missing": max(dec["req_edge"] - s["edge"], 0.0)}


def compute_early(ss: Any, cfg: Cfg, ctr: Optional[Contract], now: datetime, row: Optional[pd.Series],
                  S0: Optional[float], tau_min: Optional[float], q: dict, fresh_level: str) -> Optional[dict]:
    """SEÑAL TEMPRANA: una dirección por ventana, emitida al inicio y fijada para toda la ventana.
    Usa la tendencia (score) con más peso que el modelo normal, porque al inicio el precio está pegado
    a la referencia y la distancia casi no informa. NO exige 80–90 %: solo datos válidos y no ROJOS.
    NO es la señal BUY (esa sigue exigiendo ventaja neta) y NUNCA dispara órdenes."""
    if not cfg.early_on or ctr is None:
        return None
    store = ss.setdefault("early_calls", {})
    saved = store.get(ctr.ticker)
    elapsed = (now - ctr.window_start).total_seconds()

    live = None
    if row is not None and S0 and ctr.reference and tau_min is not None and tau_min > 0 and bool(row["valid"]):
        c2 = replace(cfg, drift_cap=max(cfg.drift_cap, cfg.early_drift))
        mp2 = model_prob(S0, ctr.reference, tau_min, float(row["sigma"]), float(row["score"]), c2)
        p_yes = mp2["p"] if ctr.yes_if_above else 1 - mp2["p"]
        side = "UP" if p_yes >= 0.5 else "DOWN"
        live = {"side": side, "p": p_yes if side == "UP" else 1 - p_yes}

    if saved is not None:
        out = dict(saved, state="emitted", live=live, elapsed=elapsed)
        out["reversal"] = bool(live and live["side"] != saved["side"] and live["p"] >= 0.60)
        if not cfg.early_lock and live:
            out["side"], out["p"] = live["side"], live["p"]
        return out
    if elapsed < cfg.early_after_s:
        return {"state": "pending", "wait_s": cfg.early_after_s - elapsed}
    if not ctr.supported or ctr.reference is None or live is None or S0 is None:
        return {"state": "blocked", "why": "faltan datos válidos del contrato o del modelo"}
    if fresh_level == "red":
        return {"state": "blocked", "why": "datos obsoletos (semáforo rojo): no se emite una dirección con datos viejos"}
    ask = q.get("yes_ask") if live["side"] == "UP" else q.get("no_ask")
    edge = None if ask is None else live["p"] - (ask + kalshi_fee(ask, cfg.fee_rate) + cfg.slippage)
    rec = {"side": live["side"], "p": live["p"], "ask": ask, "edge": edge, "btc": S0, "ref": ctr.reference,
           "at": now, "elapsed": elapsed}
    store[ctr.ticker] = rec
    hist: List[dict] = ss.history
    hist.append({"hora_utc": now.strftime("%Y-%m-%d %H:%M:%S"), "contrato": ctr.ticker, "señal": f"EARLY {live['side']}",
                 "btc": round(S0, 2), "referencia": ctr.reference, "yes_ask": q.get("yes_ask"), "no_ask": q.get("no_ask"),
                 "prob_modelo": round(live["p"], 4), "ventaja": None if edge is None else round(edge, 4),
                 "seg_restantes": round(tau_min * 60), "incertidumbre": "—", "motivo": "señal temprana",
                 "bloqueo": "", "cierre_utc": ctr.close_time.strftime("%Y-%m-%d %H:%M:%S"),
                 "resultado": None, "acierto": None})
    save_history(hist)
    return dict(rec, state="emitted", live=live, reversal=False)


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

    drift_supported = _brier(tr["p_raw"].to_numpy(), tr["y"].to_numpy()) < \
        _brier(tr["p_nd"].to_numpy(), tr["y"].to_numpy()) - 1.5e-3
    base_col = "p_raw" if drift_supported else "p_nd"

    twins = sorted(tr["window"].unique())
    cut2 = twins[int(len(twins) * 0.7)]
    t_a, t_b = tr[tr["window"] < cut2], tr[tr["window"] >= cut2]
    a0, b0 = fit_platt(t_a[base_col].to_numpy(), t_a["y"].to_numpy())
    pb = 1 / (1 + np.exp(-(a0 * logit(t_b[base_col].to_numpy()) + b0)))
    platt_ok = _brier(pb, t_b["y"].to_numpy()) < _brier(t_b[base_col].to_numpy(), t_b["y"].to_numpy()) - 1e-4
    a, b = fit_platt(tr[base_col].to_numpy(), tr["y"].to_numpy()) if platt_ok else (1.0, 0.0)

    R["p_used"] = 1 / (1 + np.exp(-(a * logit(R[base_col].to_numpy()) + b))) if platt_ok else R[base_col]
    R["p_used"] = R["p_used"].clip(0.005, 0.995)

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
# OPERACIÓN AUTOMÁTICA (órdenes en Kalshi) — APAGADA por defecto
# ──────────────────────────────────────────────────────────────────────────────
# Capas de seguridad: (1) modo Apagado/Simulación/REAL; (2) REAL exige KALSHI_LIVE_OK=1 + escribir OPERAR;
# (3) solo con semáforo VERDE y la misma señal en N lecturas seguidas; (4) órdenes límite fill-or-kill (nunca
# quedan órdenes abiertas); (5) tope por operación, por día, pérdida diaria y exposición; (6) UNA orden por
# contrato (registro en trade_log.csv + client_order_id fijo); (7) si una orden queda en duda (timeout/5xx)
# se DETIENE todo hasta que la revises en tu cuenta. Las órdenes POST nunca se reintentan solas.
# La INCLINACIÓN (lean) es solo informativa: nunca dispara órdenes.
KALSHI_TRADE_BASE = os.getenv("KALSHI_TRADE_BASE", KALSHI_BASE)
TRADE_LOG = "trade_log.csv"
TRADE_COLS = ["ts_utc", "modo", "ticker", "lado", "contratos", "precio_limite", "estado", "order_id", "fill",
              "costo", "comision", "cierre_utc", "resultado", "pnl", "error"]
_TRADE_LOCK = threading.Lock()


@dataclass
class TradeCfg:
    mode: str = "off"              # off | dry | live
    contracts: int = 1
    max_cost: float = 5.0          # $ máximo por operación (precio límite + comisión) × contratos
    max_trades_day: int = 5        # intentos REALES máximos por día (UTC)
    max_daily_loss: float = 10.0   # $ de pérdida realizada del día que detiene la operación
    max_exposure: float = 10.0     # $ máximo en operaciones aún sin resolver
    confirm_reads: int = 2         # lecturas seguidas con la MISMA señal antes de enviar
    live_unlocked: bool = False    # KALSHI_LIVE_OK=1 y confirmación escrita


def kalshi_creds() -> Tuple[Optional[str], Optional[bytes]]:
    """Claves SOLO desde variables de entorno (o st.secrets). Nunca van en el código."""
    key_id = os.getenv("KALSHI_API_KEY_ID")
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM")
    path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    try:
        if not key_id and "KALSHI_API_KEY_ID" in st.secrets:
            key_id = st.secrets["KALSHI_API_KEY_ID"]
        if not pem and not path and "KALSHI_PRIVATE_KEY_PEM" in st.secrets:
            pem = st.secrets["KALSHI_PRIVATE_KEY_PEM"]
    except Exception:
        pass
    data: Optional[bytes] = None
    if pem:
        data = pem.replace("\\n", "\n").encode()
    elif path and os.path.exists(os.path.expanduser(path)):
        with open(os.path.expanduser(path), "rb") as fh:
            data = fh.read()
    return key_id, data


def kalshi_sign(method: str, path: str, ts_ms: str, pem: bytes) -> str:
    """RSA-PSS/SHA-256 sobre  timestamp_ms + MÉTODO + ruta (con /trade-api/v2, SIN query string)."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = serialization.load_pem_private_key(pem, password=None)
    sig = key.sign(f"{ts_ms}{method.upper()}{path}".encode(),
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                   hashes.SHA256())
    return base64.b64encode(sig).decode()


def kalshi_private(method: str, path: str, body: Optional[dict] = None, timeout: Tuple[float, float] = (3.5, 8.0)) -> Any:
    """Solicitud firmada. SIN reintentos. HTTP ≥ 400 → RuntimeError('HTTP n: ...'); fallos de red → requests.RequestException."""
    key_id, pem = kalshi_creds()
    if not key_id or not pem:
        raise RuntimeError("Faltan credenciales: define KALSHI_API_KEY_ID y KALSHI_PRIVATE_KEY_PATH (o _PEM).")
    ts = str(int(time.time() * 1000))
    full = urlparse(KALSHI_TRADE_BASE).path.rstrip("/") + path
    headers = {"User-Agent": UA["User-Agent"], "Accept": "application/json", "Content-Type": "application/json",
               "KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
               "KALSHI-ACCESS-SIGNATURE": kalshi_sign(method, full, ts, pem)}
    r = get_session().request(method, KALSHI_TRADE_BASE + path, headers=headers,
                              data=None if body is None else json.dumps(body), timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:240]}")
    return r.json()


def ceil_cent(x: float) -> float:
    return min(math.ceil(x * 100 - 1e-9) / 100, 0.99)


def build_order_body(ticker: str, side_ui: str, count: int, limit_price: float, client_id: str) -> dict:
    """Orden límite de COMPRA, fill-or-kill (se ejecuta completa al instante o se cancela; no queda abierta)."""
    side = "yes" if side_ui == "UP" else "no"
    return {"ticker": ticker, "side": side, "action": "buy", "client_order_id": client_id,
            "count": int(count), "count_fp": f"{int(count):.2f}",
            ("yes_price_dollars" if side == "yes" else "no_price_dollars"): f"{limit_price:.4f}",
            "time_in_force": "fill_or_kill"}


def client_order_id_for(ticker: str) -> str:
    return "b-" + re.sub(r"[^A-Za-z0-9-]", "-", ticker)[:60]   # fijo por contrato: Kalshi rechaza duplicados


def load_trades() -> List[dict]:
    try:
        if os.path.exists(TRADE_LOG):
            return pd.read_csv(TRADE_LOG, dtype=str).replace({np.nan: None}).to_dict("records")
    except Exception:
        pass
    return []


def save_trades(rows: List[dict]) -> None:
    try:
        pd.DataFrame(rows, columns=TRADE_COLS).to_csv(TRADE_LOG, index=False)
    except Exception:
        pass


def today_stats(rows: List[dict], now: datetime) -> dict:
    day = now.strftime("%Y-%m-%d")
    live = [r for r in rows if r.get("modo") == "live"]
    attempts = [r for r in live if str(r.get("ts_utc", "")).startswith(day)]
    realized = sum(_to_float(r.get("pnl")) or 0.0 for r in live
                   if str(r.get("cierre_utc", "")).startswith(day) and r.get("pnl") not in (None, ""))
    exposure = sum((_to_float(r.get("costo")) or 0.0) + (_to_float(r.get("comision")) or 0.0) for r in live
                   if (_to_float(r.get("fill")) or 0) > 0 and not r.get("resultado"))
    return {"attempts": len(attempts), "realized": realized, "exposure": exposure}


def trade_guard(R: dict, cfg: Cfg, tcfg: TradeCfg, rows: List[dict], now: datetime,
                streak: int, halted: Optional[str]) -> Tuple[List[str], dict]:
    """Devuelve (motivos por los que NO se envía, plan). Lista vacía = se puede enviar."""
    why: List[str] = []
    plan: dict = {}
    if halted:
        why.append(f"Operación detenida: {halted}")
    if R["signal"] not in ("BUY UP", "BUY DOWN") or R["ctr"] is None or not R["dec"].get("side"):
        why.append("no hay señal BUY")
        return why, plan
    if R["fresh"]["level"] != "green":
        why.append("el semáforo de frescura no está verde")
    if streak < tcfg.confirm_reads:
        why.append(f"la señal debe repetirse {tcfg.confirm_reads} lecturas seguidas (lleva {streak})")
    ticker, side = R["ctr"].ticker, R["dec"]["side"]
    s = R["dec"]["sides"][side]
    count = int(tcfg.contracts)
    limit = ceil_cent(s["ask"])
    fee = kalshi_fee(limit, cfg.fee_rate)
    cost = count * (limit + fee)
    plan = {"ticker": ticker, "side": side, "count": count, "limit": limit, "cost": cost, "fee": fee * count}
    if cost > tcfg.max_cost:
        why.append(f"costo estimado ${cost:.2f} supera el tope por operación (${tcfg.max_cost:.2f})")
    if s["p"] - (limit + fee) < R["dec"]["req_edge"] - 1e-9:
        why.append("la ventaja al precio límite (redondeado al centavo) ya no alcanza el mínimo")
    if s.get("depth") is not None and s["depth"] < count:
        why.append(f"solo hay {s['depth']:.0f} contratos disponibles")
    if any(r.get("ticker") == ticker and r.get("modo") == tcfg.mode for r in rows):
        why.append("ya se intentó una orden en este contrato (máximo una por contrato)")
    if tcfg.mode == "live":
        st_ = today_stats(rows, now)
        if st_["attempts"] >= tcfg.max_trades_day:
            why.append(f"límite diario de {tcfg.max_trades_day} operaciones alcanzado")
        if st_["realized"] <= -tcfg.max_daily_loss:
            why.append(f"pérdida realizada del día (${-st_['realized']:.2f}) alcanzó el límite (${tcfg.max_daily_loss:.2f})")
        if st_["exposure"] + cost > tcfg.max_exposure:
            why.append(f"exposición sin resolver (${st_['exposure']:.2f}) + esta orden superaría ${tcfg.max_exposure:.2f}")
    return why, plan


def _parse_fill(order: dict) -> Tuple[float, float, float]:
    fill = _to_float(order.get("fill_count_fp"))
    if fill is None:
        fill = _to_float(order.get("fill_count")) or 0.0
    cost = _to_float(order.get("taker_fill_cost_dollars"))
    if cost is None:
        c = _to_float(order.get("taker_fill_cost"))
        cost = (c / 100.0) if c is not None else 0.0
    fees = _to_float(order.get("taker_fees_dollars"))
    if fees is None:
        f_ = _to_float(order.get("taker_fees"))
        fees = (f_ / 100.0) if f_ is not None else 0.0
    return fill, cost, fees


def auto_trade(R: dict, cfg: Cfg, tcfg: TradeCfg) -> None:
    ss = st.session_state
    sig = R["signal"]
    key = (R["ctr"].ticker if R["ctr"] else None, sig)
    ss["sig_streak"] = (ss.get("sig_streak", 0) + 1) if (sig.startswith("BUY") and ss.get("sig_key") == key) \
        else (1 if sig.startswith("BUY") else 0)
    ss["sig_key"] = key
    if tcfg.mode == "off":
        return
    if tcfg.mode == "live" and not tcfg.live_unlocked:
        ss["trade_msg"] = "Modo REAL bloqueado: define KALSHI_LIVE_OK=1 y escribe OPERAR en la barra lateral."
        return
    if not sig.startswith("BUY"):
        return
    now = R["now"]
    with _TRADE_LOCK:
        rows = load_trades()
        why, plan = trade_guard(R, cfg, tcfg, rows, now, ss["sig_streak"], ss.get("trading_halt"))
        if why:
            ss["trade_msg"] = "Orden NO enviada: " + "; ".join(why)
            return
        cid = client_order_id_for(plan["ticker"])
        row = {"ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"), "modo": tcfg.mode, "ticker": plan["ticker"],
               "lado": plan["side"], "contratos": plan["count"], "precio_limite": plan["limit"], "estado": "",
               "order_id": None, "fill": None, "costo": None, "comision": None,
               "cierre_utc": R["ctr"].close_time.strftime("%Y-%m-%d %H:%M:%S"), "resultado": None, "pnl": None,
               "error": None}
        if tcfg.mode == "dry":
            row["estado"] = "simulada"
            rows.append(row)
            save_trades(rows)
            ss["trade_msg"] = (f"SIMULACIÓN: compraría {plan['count']} × {plan['side']} en {plan['ticker']} "
                               f"a ≤ {cents(plan['limit'])} (costo ≈ ${plan['cost']:.2f}). No se envió nada.")
            return
        body = build_order_body(plan["ticker"], plan["side"], plan["count"], plan["limit"], cid)
        try:
            resp = kalshi_private("POST", "/portfolio/orders", body)
            order = resp.get("order", resp) if isinstance(resp, dict) else {}
            fill, cost, fees = _parse_fill(order)
            row.update({"order_id": order.get("order_id"), "fill": fill, "costo": cost, "comision": fees,
                        "estado": "ejecutada" if fill > 0 else "sin_ejecución"})
            ss["trade_msg"] = (f"ORDEN {row['estado'].upper()}: {plan['count']} × {plan['side']} {plan['ticker']} "
                               f"límite {cents(plan['limit'])} · llenados {fill:.0f} · costo ${cost:.2f}")
        except RuntimeError as e:
            msg = str(e)
            m_ = re.match(r"HTTP (\d+)", msg)
            if m_ and int(m_.group(1)) < 500:
                row.update({"estado": "rechazada", "error": msg[:200]})
                ss["trade_msg"] = "Kalshi RECHAZÓ la orden: " + msg[:200]
            else:
                row.update({"estado": "INCIERTA", "error": msg[:200]})
                ss["trading_halt"] = "una orden quedó en duda (error del servidor); revisa tu cuenta de Kalshi y reinicia la app"
                ss["trade_msg"] = "⚠️ Resultado de la orden INCIERTO. Operación detenida. Revisa tu cuenta de Kalshi: " + msg[:160]
        except requests.RequestException as e:
            row.update({"estado": "INCIERTA", "error": f"{type(e).__name__}"})
            ss["trading_halt"] = "una orden quedó en duda (fallo de red); revisa tu cuenta de Kalshi y reinicia la app"
            ss["trade_msg"] = "⚠️ Resultado de la orden INCIERTO (fallo de red). Operación detenida. Revisa tu cuenta de Kalshi."
        except Exception as e:  # noqa: BLE001  (p. ej. falta 'cryptography' o clave inválida: la orden NO salió)
            row.update({"estado": "no_enviada", "error": f"{type(e).__name__}: {str(e)[:160]}"})
            ss["trade_msg"] = f"No se envió la orden: {type(e).__name__}: {str(e)[:160]}"
        rows.append(row)
        save_trades(rows)


def resolve_trades(now: datetime) -> None:
    """Calcula el resultado y el P&L real de órdenes REALES ya vencidas (consulta a Kalshi, máx. 2 por pasada)."""
    with _TRADE_LOCK:
        rows = load_trades()
        done, cache = 0, {}
        for r in rows:
            if r.get("modo") != "live" or r.get("resultado") or (_to_float(r.get("fill")) or 0) <= 0:
                continue
            try:
                close = pd.to_datetime(r["cierre_utc"], utc=True).to_pydatetime()
            except Exception:
                continue
            if now < close + timedelta(seconds=90) or done >= 2:
                continue
            t = r["ticker"]
            if t not in cache:
                try:
                    cache[t] = fetch_kalshi_market_result(t)
                except Exception:
                    cache[t] = None
                done += 1
            res = cache[t]
            if res:
                won = (r["lado"] == "UP" and res == "yes") or (r["lado"] == "DOWN" and res == "no")
                fill, cost, fees = (_to_float(r.get("fill")) or 0), (_to_float(r.get("costo")) or 0), (_to_float(r.get("comision")) or 0)
                r["resultado"], r["pnl"] = res, round((fill - cost - fees) if won else -(cost + fees), 4)
        if done:
            save_trades(rows)


def render_trading(R: dict, cfg: Cfg, tcfg: TradeCfg) -> None:
    ss = st.session_state
    label = {"off": "APAGADA", "dry": "SIMULACIÓN (no envía nada)", "live": "REAL (envía órdenes)"}[tcfg.mode]
    key_id, pem = kalshi_creds()
    try:
        import cryptography  # noqa: F401
        crypto_ok = True
    except Exception:
        crypto_ok = False
    rows = load_trades()
    with st.expander(f"🤖 Operación automática · {label}", expanded=(tcfg.mode != "off")):
        if ss.get("trading_halt"):
            st.markdown(f"<div class='why'>⛔ <b>Operación detenida</b><br>{esc(ss['trading_halt'])}</div>", unsafe_allow_html=True)
        if tcfg.mode == "live" and not tcfg.live_unlocked:
            st.markdown("<div class='why'>🔒 Modo REAL bloqueado hasta definir <b>KALSHI_LIVE_OK=1</b> y escribir "
                        "<b>OPERAR</b> en la barra lateral.</div>", unsafe_allow_html=True)
        if ss.get("trade_msg"):
            st.markdown(f"<div class='note'>{esc(ss['trade_msg'])}</div>", unsafe_allow_html=True)
        stt = today_stats(rows, R["now"])
        st.markdown(kv_html([
            ("Entorno de órdenes", KALSHI_TRADE_BASE), ("Credenciales", "cargadas" if (key_id and pem) else "NO definidas"),
            ("Librería 'cryptography'", "instalada" if crypto_ok else "falta (pip install cryptography)"),
            ("Contratos por orden / tope por orden", f"{tcfg.contracts} / ${tcfg.max_cost:.2f}"),
            ("Intentos reales hoy (UTC)", f"{stt['attempts']} / {tcfg.max_trades_day}"),
            ("P&L realizado hoy", f"${stt['realized']:+.2f} (límite −${tcfg.max_daily_loss:.2f})"),
            ("Exposición sin resolver", f"${stt['exposure']:.2f} (máx. ${tcfg.max_exposure:.2f})"),
            ("Lecturas seguidas de la señal", f"{ss.get('sig_streak', 0)} / {tcfg.confirm_reads}")]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Probar credenciales"):
            try:
                b = kalshi_private("GET", "/portfolio/balance")
                bal = b.get("balance_dollars") if isinstance(b, dict) and b.get("balance_dollars") is not None else \
                    (None if not isinstance(b, dict) or b.get("balance") is None else float(b["balance"]) / 100)
                st.success(f"Conexión firmada correcta. Saldo: {fnum(bal, 2, '$')}" if bal is not None else
                           "Conexión firmada correcta (respuesta sin saldo reconocible).")
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {str(e)[:200]}")
        if c2.button("⛔ Detener operación"):
            ss["trading_halt"] = "detenida manualmente (reinicia la app para reanudar)"
            st.warning("Operación detenida.")
        if rows:
            last = [[r["ts_utc"][11:19] if r.get("ts_utc") else "—", r.get("modo"), str(r.get("ticker", "")).replace(SERIES + "-", ""),
                     r.get("lado"), r.get("estado"), fnum(_to_float(r.get("pnl")), 2, "$") if r.get("pnl") else "—"]
                    for r in rows[::-1][:8]]
            st.markdown(table_html(["Hora UTC", "Modo", "Contrato", "Lado", "Estado", "P&L"], last), unsafe_allow_html=True)
            st.download_button("Descargar registro de operaciones (CSV)", pd.DataFrame(rows, columns=TRADE_COLS).to_csv(index=False).encode("utf-8"),
                               file_name="trade_log.csv", mime="text/csv")
        st.caption("Las órdenes son límite fill-or-kill: se ejecutan completas al instante o se cancelan. "
                   "Una señal es una estimación, no una garantía: puedes perder lo que operes.")


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
    ss.setdefault("data_log", {"refresh_n": 0, "last_candle_start": None, "last_new_candle_at": None,
                               "last_price": None, "last_price_change_at": None})


def safe_fetch(name: str, fn: Callable[[], dict], min_s: float) -> Tuple[Optional[dict], bool]:
    """Devuelve (datos, último_intento_falló).

    * Hay una descarga REAL como máximo cada `min_s` segundos; entre medias solo se redibuja la pantalla con la
      última lectura buena, que conserva su hora de descarga ORIGINAL (nunca se hace pasar por reciente).
    * Si falla, se conserva la última lectura buena y su antigüedad real la hará bloquear la señal."""
    ss = st.session_state
    t = time.time()
    prev = ss.status.get(name, {})
    if prev.get("last_try") and t - prev["last_try"] < min_s and name in ss.last_good:
        return ss.last_good[name], not prev.get("ok", False)
    try:
        out = fn()
        ss.last_good[name] = out
        ss.status[name] = {"ok": True, "error": None, "ok_at": out["fetched_at"], "latency_ms": out.get("latency_ms"),
                           "last_try": t, "fails": 0, "n_ok": prev.get("n_ok", 0) + 1}
        return out, False
    except Exception as e:  # noqa: BLE001
        ss.status[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:220]}", "ok_at": prev.get("ok_at"),
                           "latency_ms": None, "last_try": t, "fails": prev.get("fails", 0) + 1,
                           "n_ok": prev.get("n_ok", 0)}
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
[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .45rem !important;}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="column"] {min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;}
div[data-testid="stMetric"] {background:#1a1530;border:1px solid #f59e0b55;border-radius:12px;padding:7px 10px;}
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
.lean {margin-top:8px;padding:.4rem .6rem;border-radius:10px;background:#1d1838;border:1px dashed #6d5bd0;
       color:#e4defa;font-size:.82rem;line-height:1.35;text-align:left;}
.lean b.up {color:#4ade80;} .lean b.down {color:#f87171;}
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
.fresh {border:2px solid;border-radius:14px;padding:.5rem .7rem;margin:.5rem 0 .3rem;background:#161129;}
.fresh.green {border-color:#22c55e;} .fresh.yellow {border-color:#f59e0b;} .fresh.red {border-color:#ef4444;}
.fresh .t {font-weight:700;font-size:.9rem;color:#ece8fa;}
.dot {display:inline-block;width:.8rem;height:.8rem;border-radius:50%;margin-right:.45rem;vertical-align:-1px;}
.dot.green {background:#22c55e;box-shadow:0 0 8px #22c55e;} .dot.yellow {background:#f59e0b;box-shadow:0 0 8px #f59e0b;}
.dot.red {background:#ef4444;box-shadow:0 0 8px #ef4444;}
table.kv, table.tb {width:100%;border-collapse:collapse;font-size:.8rem;margin:.2rem 0 .5rem;}
table.kv td {padding:3px 4px;border-bottom:1px solid #2b2150;vertical-align:top;}
table.kv td.k {color:#a78bfa;width:52%;} table.kv td.v {text-align:right;color:#f1edfb;word-break:break-word;}
table.tb th {color:#a78bfa;font-weight:600;text-align:right;padding:3px 4px;border-bottom:1px solid #3a2d70;}
table.tb th:first-child, table.tb td:first-child {text-align:left;}
table.tb td {padding:3px 4px;border-bottom:1px solid #2b2150;text-align:right;color:#f1edfb;}
table.tb td:first-child {color:#a78bfa;}
</style>
"""


def signal_card(signal: str, subtitle: str, lean_html: str = "") -> str:
    color = {"BUY UP": "#22c55e", "BUY DOWN": "#ef4444", "ESPERAR": "#f59e0b"}.get(signal, "#a78bfa")
    return (f'<div class="sig" style="border-color:{color}"><div class="sig-label">SEÑAL</div>'
            f'<div class="sig-main" style="color:{color}">{esc(signal)}</div>'
            f'<div class="sig-sub">{esc(subtitle)}</div>{lean_html}</div>')


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
    ss = st.session_state
    now = utcnow()
    dl = ss.data_log
    dl["refresh_n"] += 1
    tick, tick_fail = safe_fetch("Coinbase · precio", fetch_ticker, 3)
    cand, cand_fail = safe_fetch("Coinbase · velas 1m", lambda: fetch_candles(cfg.stale_trigger_s), 5)
    kal, kal_fail = safe_fetch("Kalshi · mercados", fetch_kalshi_markets, 4)
    stat = ss.status
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
    if tick is None or a > cfg.max_age_s:
        add("price", "Precio BTC (Coinbase)", "fail", "sin datos" if tick is None else f"dato de hace {a:.0f} s",
            lim_age, "Precio de Coinbase no disponible o desactualizado." + err_of("Coinbase · precio"))
    elif tick_fail:
        add("price", "Precio BTC (Coinbase)", "warn", f"{fnum(tick['price'], 2, '$')} · hace {a:.1f} s", lim_age,
            "El último intento falló; se usa la lectura anterior (aún dentro del límite)."
            + err_of("Coinbase · precio"), blocking=False)
    else:
        add("price", "Precio BTC (Coinbase)", "ok", f"{fnum(tick['price'], 2, '$')} · hace {a:.1f} s", lim_age)
        if dl["last_price"] != tick["price"]:
            dl["last_price"], dl["last_price_change_at"] = tick["price"], tick["fetched_at"]

    # ── 2. Hora de la última operación de Coinbase ──
    ex = None if tick is None else tick.get("exchange_time")
    ta = None
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
    n_gap, lag, gaps = 0, None, (0, 0)
    if cand is not None:
        closed_raw, fm = split_closed(cand["df"], now)
        if len(closed_raw):
            gaps = gap_stats(closed_raw)
            closed, n_gap = fill_gaps(closed_raw)
            lag = candle_age_s(closed, now)
            f = compute_features(closed)
            row = f.iloc[-1]
            last_start = closed["time"].iloc[-1].to_pydatetime()
            if dl["last_candle_start"] != last_start:       # llegó una vela cerrada NUEVA (no solo se redibujó)
                dl["last_candle_start"], dl["last_new_candle_at"] = last_start, cand["fetched_at"]
        if len(fm):
            forming = fm.iloc[-1]
    src = "" if cand is None else cand.get("source", "")
    lim_c = f"antigüedad ≤ {cfg.candle_max_s} s"
    if cand is None:
        add("candles", "Velas de 1 min (Coinbase)", "fail", "sin datos", lim_c,
            "Velas de Coinbase no disponibles." + err_of("Coinbase · velas 1m"))
    elif closed is None:
        add("candles", "Velas de 1 min (Coinbase)", "fail", "0 velas cerradas", lim_c,
            "Coinbase no devolvió ninguna vela cerrada.")
    elif lag > cfg.candle_max_s:
        add("candles", "Velas de 1 min (Coinbase)", "fail", f"última vela cerró hace {lag:.0f} s · {src}", lim_c,
            f"La última vela cerrada de Coinbase ({loc(closed['time'].iloc[-1].to_pydatetime())}) tiene {lag:.0f} s "
            f"de antigüedad (máximo {cfg.candle_max_s} s). Ninguna fuente entregó datos más recientes: las señales "
            "que dependen de ella quedan bloqueadas. Revisa el panel de diagnóstico (fuentes de velas)."
            + err_of("Coinbase · velas 1m"))
    elif cand_fail:
        add("candles", "Velas de 1 min (Coinbase)", "warn", f"última vela cerró hace {max(lag, 0):.0f} s · {src}", lim_c,
            "El último intento de descarga falló; se usan las velas anteriores (aún dentro del límite)."
            + err_of("Coinbase · velas 1m"), blocking=False)
    else:
        add("candles", "Velas de 1 min (Coinbase)", "ok",
            f"{len(closed)} velas · última cerró hace {max(lag, 0):.0f} s · {src}", lim_c)
    if closed is not None:
        g_all, g_10 = gaps
        if g_10 > 2:
            add("gaps", "Velas faltantes recientes", "fail", f"{g_10} faltan en los últimos 10 min", "≤ 2",
                "Coinbase omitió varios minutos recientes: los indicadores no son fiables.")
        else:
            add("gaps", "Velas faltantes recientes", "ok" if g_all == 0 else "warn",
                f"{g_all} huecos en el tramo ({g_10} en los últimos 10 min)", "≤ 2 recientes",
                "Minutos sin operaciones: se rellenan planos (volumen 0) solo dentro del tramo." if g_all else "",
                blocking=False)
        rep = (cand or {}).get("report") or {}
        bad = {k: v for k, v in rep.items() if k != "filas" and v}
        add("candle_valid", "Validación de velas recibidas", "ok" if not bad else "warn",
            f"{rep.get('filas', 0)} filas" + ("" if not bad else " · " + ", ".join(f"{k}: {v}" for k, v in bad.items())),
            "", "Se descartaron/ordenaron filas inválidas, duplicadas, desalineadas o futuras." if bad else "",
            blocking=False)
    if f is None:
        add("warmup", "Indicadores calculados", "na", blocking=False)
    else:
        w_ok = len(f) >= WARMUP and bool(row["valid"])
        add("warmup", "Indicadores calculados", "ok" if w_ok else "fail", f"{len(f)} velas", f"≥ {WARMUP} y sin NaN",
            "" if w_ok else f"Datos insuficientes para los indicadores (se necesitan ≥ {WARMUP} velas cerradas válidas).")
    add("forming", "Vela actual de Coinbase", "ok" if forming is not None else "warn",
        "presente" if forming is not None else "ausente", "",
        "Coinbase aún no publicó la vela del minuto en curso. Solo afecta a la gráfica; no bloquea la señal.",
        blocking=False)

    # ── 4. Kalshi: frescura ──
    a_k = age_of(kal)
    if kal is None or a_k > cfg.max_age_s:
        add("kalshi", "Datos de Kalshi", "fail", "sin datos" if kal is None else f"dato de hace {a_k:.0f} s", lim_age,
            "Datos de Kalshi no disponibles o desactualizados." + err_of("Kalshi · mercados"))
    elif kal_fail:
        add("kalshi", "Datos de Kalshi", "warn", f"descargados hace {a_k:.1f} s", lim_age,
            "El último intento falló; se usa la lectura anterior (aún dentro del límite)."
            + err_of("Kalshi · mercados"), blocking=False)
    else:
        add("kalshi", "Datos de Kalshi", "ok", f"descargados hace {a_k:.1f} s", lim_age)

    # ── 5. Reloj local (hora de descarga vs. cabecera Date del servidor) ──
    signed = []
    for o in (tick, cand, kal):
        sd_ = o.get("server_date") if o else None
        if sd_ is not None:
            signed.append((o["fetched_at"] - sd_).total_seconds())
    if signed:
        skew = min(signed)
        sk_ok = abs(skew) <= cfg.max_clock_skew_s
        add("clock", "Reloj del dispositivo/servidor", "ok" if sk_ok else "fail", f"desfase ≈ {skew:+.1f} s",
            f"±{cfg.max_clock_skew_s:.0f} s",
            "" if sk_ok else f"El reloj de la máquina que ejecuta la app difiere ~{abs(skew):.0f} s de los servidores; "
            "sincronízalo (un reloj adelantado/atrasado falsea la antigüedad de las velas).")
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
    if S0 is not None and closed is not None:
        dv = abs(math.log(S0 / float(closed["close"].iloc[-1])))
        add("price_vs_candle", "Precio actual vs última vela", "ok" if dv <= 0.01 else "warn", f"{dv * 100:.2f}% de diferencia",
            "≤ 1%", "El precio en vivo y la última vela cerrada discrepan: posible dato atrasado (no bloquea)."
            if dv > 0.01 else "", blocking=False)

    # ── 7. Coherencia de precios de Kalshi ──
    q: Dict[str, Optional[float]] = {k: None for k in PRICE_KEYS}
    q["yes_ask_size"], q["yes_bid_size"] = None, None
    if mk is not None:
        q = {k: mk[k] for k in PRICE_KEYS}
        q["yes_ask_size"], q["yes_bid_size"] = mk.get("yes_ask_size"), mk.get("yes_bid_size")
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
    alerts: List[Tuple[str, int]] = []
    if (row is not None and ctr is not None and ctr.reference and S0 and tau_min is not None
            and tau_min > 0 and bool(row["valid"])):
        bt = ss.bt["validated"] if (cfg.use_backtest and ss.bt) else None
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
        alerts = turn_alerts(f)
        lean_btc = 1 if mp["p"] >= 0.5 else -1
        turn_against = sum(1 for _, dr in alerts if dr == -lean_btc)
        spreads = [x - y for x, y in ((q["yes_ask"], q["yes_bid"]), (q["no_ask"], q["no_bid"]))
                   if x is not None and y is not None]
        unc, why_unc, unc_pts = uncertainty(mp["z"], mp["sd"], float(row["sigma_ratio"]),
                                            min(spreads) if spreads else None, tau_min, calibrated, cfg, turn_against)

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
    lean = compute_lean(p_final, dec)
    all_filters = F + dec["filters"]
    blockers = [f_ for f_ in all_filters if f_["status"] == "fail" and f_["blocking"]]

    # ── 10. Semáforo de frescura ──
    conn_err = [f"{n}: {(stat.get(n) or {}).get('error')}" for n in
                ("Coinbase · precio", "Coinbase · velas 1m", "Kalshi · mercados")
                if (stat.get(n) or {}).get("ok") is False]
    fresh_level, fresh_why = freshness(age_of(tick), ta, lag, a_k, conn_err, cfg)
    early = compute_early(ss, cfg, ctr, now, row, S0, tau_min, q, fresh_level)
    return {"now": now, "cfg": cfg, "forced": forced, "tick": tick, "cand": cand, "kal": kal, "mk": mk, "ctr": ctr,
            "open_markets": open_markets, "S0": S0, "tau_min": tau_min, "f": f, "closed": closed, "forming": forming,
            "row": row, "n_gap": n_gap, "lag": lag, "mp": mp, "p_final": p_final, "p_model_yes": p_model_yes,
            "p_nd_yes": p_nd_yes, "platt": platt_used, "unc": unc, "why_unc": why_unc, "unc_pts": unc_pts,
            "calibrated": calibrated, "notes_model": notes_model, "q": q, "implied": implied, "dec": dec,
            "signal": dec["signal"], "lean": lean, "early": early, "filters": all_filters, "blockers": blockers, "alerts": alerts,
            "fresh": {"level": fresh_level, "why": fresh_why}, "data_log": dict(dl)}


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
            ("Modelo calibrado", "Sí" if R["calibrated"] else "No (margen extra y +1 pt de incertidumbre)"),
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
        ("Pantalla refrescada (local)", loc(R["now"], "%Y-%m-%d %H:%M:%S")), ("Pantalla refrescada (UTC)", R["now"].strftime("%H:%M:%S"))]
    if tick is not None:
        times += [("Precio descargado", loc(tick["fetched_at"])),
                  ("Última operación Coinbase", loc(tick.get("exchange_time"))),
                  ("Latencia Coinbase precio", fnum(tick.get("latency_ms"), 0, "", " ms"))]
    if R["closed"] is not None:
        last_open = R["closed"]["time"].iloc[-1].to_pydatetime()
        times += [("Última vela cerrada (inicio)", loc(last_open)), ("Última vela cerrada (cierre)", loc(last_open + timedelta(seconds=60)))]
    if R["kal"] is not None:
        times.append(("Kalshi descargado", loc(R["kal"]["fetched_at"])))
    if mk is not None:
        times += [("Kalshi: updated_time", loc(mk.get("updated"))),
                  ("Ventana: inicio", loc(ctr.window_start if ctr else None)), ("Ventana: cierre del CONTRATO", loc(mk["close_time"]))]
    return {"probs": probs, "read": read, "ind": ind, "times": times}


def sources_rows(R: dict) -> List[List[str]]:
    cand = R["cand"]
    if cand is None:
        return []
    rows = []
    for a in cand.get("attempts", []):
        used = "★ " if a["source"] == cand.get("source") else ""
        if a["ok"]:
            rows.append([used + a["source"], "OK", fnum(a["candle_age_s"], 0, "", " s"),
                         fnum(a["resp_age_s"], 1, "", " s"), fnum(a["age_header"], 0, "", " s") if a["age_header"] is not None else "—",
                         str(a["cache"] or "—"), fnum(a["latency_ms"], 0, "", " ms")])
        else:
            rows.append([used + a["source"], "ERROR", "—", "—", "—", "—", str(a["error"])[:60]])
    return rows


SRC_HDR = ["Fuente", "Estado", "Edad vela", "Edad resp.", "Age hdr", "Caché", "Latencia/Error"]


def sides_html(R: dict, cfg: Cfg) -> str:
    dec = R["dec"]
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
        ["Contratos al precio"] + [g(k, lambda s: fnum(s["depth"], 0)) for k in ("UP", "DOWN")],
        ["Spread"] + [g(k, lambda s: cents(s["spread"])) for k in ("UP", "DOWN")],
        ["Comisión"] + [g(k, lambda s: f"{s['fee'] * 100:.2f}¢") for k in ("UP", "DOWN")],
        ["Deslizamiento"] + [g(k, lambda s: cents(cfg.slippage)) for k in ("UP", "DOWN")],
        ["Costo equilibrio"] + [g(k, lambda s: pct(s["breakeven"])) for k in ("UP", "DOWN")],
        ["Prob. mínima p/ señal"] + [g(k, lambda s: pct(s["req_p"]) + (" ⛔" if s["req_p"] > 0.995 else ""))
                                     for k in ("UP", "DOWN")],
        ["Ventaja neta"] + [g(k, lambda s: f"{s['edge'] * 100:+.1f}¢") for k in ("UP", "DOWN")],
        ["Ventaja exigida", f"{dec['req_edge'] * 100:.1f}¢", f"{dec['req_edge'] * 100:.1f}¢"],
    ]
    hdr = ["", "UP (YES)" + (" ★" if best == "UP" else ""), "DOWN (NO)" + (" ★" if best == "DOWN" else "")]
    return table_html(hdr, rows)


def edge_explanation(R: dict, cfg: Cfg) -> str:
    dec = R["dec"]
    if not dec.get("best") or dec.get("req_edge") is None:
        return ""
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
        txt += (f" Como el modelo no está validado, se suman {cfg.uncal_extra * 100:.1f}¢ de margen "
                "(ejecuta el Backtest para validarlo; no se activa solo).")
    return txt


def build_diag_text(R: dict, cfg: Cfg) -> str:
    L = [f"DIAGNÓSTICO {loc(R['now'], '%Y-%m-%d %H:%M:%S')} local / {R['now'].strftime('%H:%M:%S')} UTC",
         f"Señal: {R['signal']}  ·  Frescura: {R['fresh']['level']}",
         f"Contrato: {R['ctr'].ticker if R['ctr'] else '—'}  cierre: {loc(R['mk']['close_time']) if R['mk'] else '—'}"]
    if R.get("lean"):
        ln = R["lean"]
        L.append(f"Inclinación: {ln['side']} {pct(ln['p'])} · ventaja {ln['edge'] * 100:+.1f}¢ · faltan {ln['missing'] * 100:.1f}¢")
    L += ["", "FILTROS"]
    for f_ in R["filters"]:
        tag = {"ok": "OK   ", "fail": "FALLA", "warn": "AVISO", "na": "N/D  "}[f_["status"]]
        L.append(f"[{tag}] {f_['name']}: {f_['value']}" + (f" | límite {f_['limit']}" if f_["limit"] else "")
                 + (f" | {f_['detail']}" if f_["detail"] and f_["status"] in ("fail", "warn") else ""))
    sr = sources_rows(R)
    if sr:
        L += ["", "FUENTES DE VELAS (" + " | ".join(SRC_HDR) + ")"] + [" | ".join(r) for r in sr]
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
        changed = True
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
    pending = [h for h in hist if str(h.get("señal", "")).startswith(("BUY", "EARLY")) and not h.get("resultado")]
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
            h["acierto"] = bool((h["señal"] in ("BUY UP", "EARLY UP") and res == "yes")
                            or (h["señal"] in ("BUY DOWN", "EARLY DOWN") and res == "no"))
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
                        f"<div class='fm'>{s['latency_ms']:.0f} ms · última respuesta correcta {loc(s['ok_at'])} · "
                        f"{s.get('n_ok', 0)} descargas</div></div>")
        else:
            extra = (f"último dato bueno ANTIGUO ({age:.0f} s): {loc(s['ok_at'])}" if age is not None else "sin datos previos")
            rows.append(f"<div class='frow fail'><div class='fh'>{esc(name)} <span class='pill bad'>ERROR ×{s.get('fails', 1)}</span></div>"
                        f"<div class='fm'>{esc(extra)}</div><div class='fd'>{esc(s['error'])}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def lean_block(R: dict) -> str:
    """Bloque HTML con la inclinación del modelo (solo cuando la señal es ESPERAR y hay datos suficientes)."""
    ln = R.get("lean")
    if not ln or R["signal"] != "ESPERAR":
        return ""
    cls = "up" if ln["side"] == "UP" else "down"
    if ln["missing"] <= 1e-9:
        gap = "la ventaja ya alcanza, pero otro filtro bloquea (ver abajo)"
    else:
        gap = f"faltan <b>{ln['missing'] * 100:.1f}¢</b> de ventaja neta para operar"
    return (f"<div class='lean'>🧭 Inclinación: <b class='{cls}'>{ln['side']}</b> · modelo {pct(ln['p'])} · "
            f"ventaja actual {ln['edge'] * 100:+.1f}¢ · {gap}.<br>"
            f"<span style='color:#9a90c4'>Es solo una lectura informativa: <b>no es una señal de compra</b>.</span></div>")


def render_early(R: dict) -> None:
    e = R.get("early")
    if e is None:
        return
    if e["state"] == "pending":
        st.markdown(f"<div class='note'>⏱ Señal temprana de esta ventana en ~{max(e['wait_s'], 0):.0f} s.</div>",
                    unsafe_allow_html=True)
        return
    if e["state"] == "blocked":
        st.markdown(f"<div class='why'>⏸ Señal temprana no emitida: {esc(e['why'])}.</div>", unsafe_allow_html=True)
        return
    color = "#22c55e" if e["side"] == "UP" else "#ef4444"
    word = "SUBE · comprar YES" if e["side"] == "UP" else "BAJA · comprar NO"
    ask = "—" if e.get("ask") is None else cents(e["ask"])
    edge = "—" if e.get("edge") is None else f"{e['edge'] * 100:+.1f}¢"
    rev = ""
    if e.get("reversal") and e.get("live"):
        rev = (f"<div class='lean'>⚠️ Ahora el modelo apunta a <b>{e['live']['side']}</b> ({pct(e['live']['p'])}). "
               "La señal temprana sigue fijada; úsalo como aviso de posible giro.</div>")
    st.markdown(
        f"<div class='sig' style='border-color:{color};margin-top:8px'><div class='sig-label'>SEÑAL TEMPRANA · ESTA VENTANA</div>"
        f"<div class='sig-main' style='color:{color}'>{esc(e['side'])}</div>"
        f"<div class='sig-sub'>{word} · emitida a los {e['elapsed']:.0f} s · modelo {pct(e['p'])}<br>"
        f"precio de compra {ask} · ventaja neta {edge} tras comisión</div>{rev}</div>"
        "<div class='fm' style='margin:.3rem 0'>Es una dirección estimada, no una garantía. Una ventaja neta negativa "
        "significa que, a ese precio, el mercado ya cobra esa probabilidad.</div>", unsafe_allow_html=True)


def render_header(R: dict, cfg: Cfg) -> None:
    ctr, tau = R["ctr"], R["tau_min"]
    sub = f"{ctr.ticker} · quedan {fmt_left(tau * 60)}" if (ctr is not None and tau is not None) else "Sin contrato analizable"
    st.markdown(signal_card(R["signal"], sub, lean_block(R)), unsafe_allow_html=True)
    if R["signal"] == "ESPERAR" and R["blockers"]:
        b = R["blockers"][0]
        extra = f"  (+{len(R['blockers']) - 1} bloqueo(s) más)" if len(R["blockers"]) > 1 else ""
        st.markdown(f"<div class='why'>⛔ <b>{esc(b['name'])}</b>{esc(extra)}<br>{esc(b['detail'])}</div>",
                    unsafe_allow_html=True)


def render_freshness(R: dict, cfg: Cfg) -> None:
    """Semáforo + datos reales de actualización. 'Pantalla refrescada' ≠ 'datos nuevos del mercado'."""
    lvl, why = R["fresh"]["level"], R["fresh"]["why"]
    title = {"green": "Datos frescos", "yellow": "Datos con retraso leve", "red": "Datos obsoletos o sin conexión"}[lvl]
    st.markdown(f"<div class='fresh {lvl}'><span class='dot {lvl}'></span><span class='t'>{title}</span>"
                + "".join(f"<div class='fm'>• {esc(w)}</div>" for w in why) + "</div>", unsafe_allow_html=True)
    now, tick, cand, closed, dl = R["now"], R["tick"], R["cand"], R["closed"], R["data_log"]
    stt = st.session_state.status
    last_ok = [s["ok_at"] for s in stt.values() if s.get("ok_at") is not None]
    cb = [stt.get(n, {}).get("ok_at") for n in ("Coinbase · precio", "Coinbase · velas 1m")]
    cb_ok = max([x for x in cb if x is not None], default=None)
    n_err = sum(1 for n in ("Coinbase · precio", "Coinbase · velas 1m", "Kalshi · mercados") if stt.get(n, {}).get("ok") is False)
    conn = "Todas OK" if (n_err == 0 and len(last_ok) == 3) else f"{n_err} con error" if n_err else "esperando datos"
    rows: List[Tuple[str, str]] = [("BTC/USD (Coinbase)", "—" if R["S0"] is None else fnum(R["S0"], 2, "$"))]
    if closed is not None:
        st_ = closed["time"].iloc[-1].to_pydatetime()
        rows += [("Última vela cerrada (local)", f"{loc(st_, '%H:%M')} → {loc(st_ + timedelta(seconds=60), '%H:%M')}"),
                 ("Antigüedad real de esa vela", f"{max(R['lag'], 0):.0f} s (máx. {cfg.candle_max_s} s)")]
    else:
        rows += [("Última vela cerrada", "—"), ("Antigüedad real de esa vela", "sin datos")]
    rows += [("Última respuesta correcta de Coinbase", "—" if cb_ok is None else f"{loc(cb_ok)} (hace {(now - cb_ok).total_seconds():.0f} s)"),
             ("Fuente de las velas", "—" if cand is None else cand.get("source", "—")),
             ("Estado de la conexión", conn),
             ("Pantalla refrescada", f"#{dl['refresh_n']} · {loc(now)} (cada {cfg.refresh_s} s)"),
             ("Última vela NUEVA recibida", "—" if dl["last_new_candle_at"] is None else
              f"{loc(dl['last_new_candle_at'])} (hace {(now - dl['last_new_candle_at']).total_seconds():.0f} s)"),
             ("Último precio distinto recibido", "—" if dl["last_price_change_at"] is None else
              f"{loc(dl['last_price_change_at'])} (hace {(now - dl['last_price_change_at']).total_seconds():.0f} s)")]
    st.markdown(kv_html(rows), unsafe_allow_html=True)
    st.caption(f"Zona horaria: {LOCAL_TZ.key} · {now.strftime('%H:%M:%S')} UTC. Redibujar la pantalla no es recibir datos "
               "nuevos: mira «Última vela NUEVA recibida».")


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
        sr = sources_rows(R)
        if sr:
            st.markdown("**Fuentes de velas** (★ = la usada; «Edad resp.» alta o «Age hdr» > 0 indica copia en caché)")
            st.markdown(table_html(SRC_HDR, sr), unsafe_allow_html=True)
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
        if R["alerts"]:
            st.markdown("**Posibles cambios de dirección / volatilidad**")
            st.markdown("".join(f"<div class='fm'>• {esc(t)}</div>" for t, _ in R["alerts"]), unsafe_allow_html=True)
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
        for t, dr in R["alerts"]:
            if dr == -dir_btc:
                con.append("Posible giro: " + t)
            elif dr == dir_btc:
                pro.append("Refuerzo: " + t)
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
                       f"Incluye deslizamiento supuesto de {cfg.slippage * 100:.1f}¢.")
        st.caption("Estimación estadística, no una certeza: ninguna señal garantiza ganancias.")


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
            ("Inicio de ventana", loc(ctr.window_start)), ("Cierre del contrato", loc(ctr.close_time)),
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


def render_live(cfg: Cfg, forced: Optional[str], tcfg: Optional[TradeCfg] = None) -> None:
    tcfg = tcfg or TradeCfg()
    R = analyze(cfg, forced)
    record_wait(R)
    auto_trade(R, cfg, tcfg)
    render_header(R, cfg)
    render_early(R)
    render_freshness(R, cfg)
    render_metrics(R)
    render_trading(R, cfg, tcfg)
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
    if tcfg.mode == "live":
        resolve_trades(now)
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
    ear = h[h["señal"].astype(str).str.startswith("EARLY")]
    ear_r = ear[ear["resultado"].notna() & (ear["resultado"] != "")]
    ke = int((ear_r["acierto"].astype(str) == "True").sum())
    st.caption(f"Señales tempranas: {len(ear)} · resueltas: {len(ear_r)} · aciertos: {ke}"
               + (f" ({ke / len(ear_r) * 100:.0f}%)" if len(ear_r) else ""))
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


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-PRUEBAS (datos SINTÉTICOS: solo para comprobar la lógica; la app nunca los usa)
# ──────────────────────────────────────────────────────────────────────────────
def _synth_raw(n: int, end_start: datetime, seed: int = 7, price0: float = 100_000.0) -> List[List[float]]:
    """n velas sintéticas, la última con inicio = end_start. Formato Coinbase (más reciente primero)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 4e-4, n)
    closes = price0 * np.exp(np.cumsum(rets))
    rows = []
    for i in range(n):
        t = end_start - timedelta(minutes=(n - 1 - i))
        o = closes[i - 1] if i else price0
        c = closes[i]
        hi, lo = max(o, c) * 1.0002, min(o, c) * 0.9998
        rows.append([float(int(t.timestamp())), lo, hi, o, c, float(rng.uniform(1, 5))])
    return rows[::-1]


def selftest() -> List[Tuple[str, bool, str]]:
    T: List[Tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        T.append((name, bool(cond), detail))

    now = datetime(2026, 10, 23, 12, 0, 30, tzinfo=timezone.utc)
    cfg = Cfg()

    # 1) parse_candles: inválidas, duplicadas, desalineadas, futuras, desordenadas
    base = int(datetime(2026, 10, 23, 11, 57, tzinfo=timezone.utc).timestamp())
    raw = [[base + 120, 99, 101, 100, 100.5, 1], [base, 99, 101, 100, 100.5, 1], [base + 60, 99, 101, 100, 100.2, 1],
           [base + 60, 99, 101, 100, 100.3, 1],                   # duplicada
           [base + 90, 99, 101, 100, 100.5, 1],                   # desalineada
           [base + 180, 99, 101, 100, 100.5, 1],                  # 12:00:00 → válida y en curso
           [base + 600, 99, 101, 100, 100.5, 1],                  # futura
           [base + 240, 105, 101, 100, 100.5, 1]]                 # OHLC inválido (low > high)
    df, rep = parse_candles(raw, now)
    chk("parse: descarta inválidas/desalineadas/futuras/duplicadas",
        rep["invalidas"] == 1 and rep["desalineadas"] == 1 and rep["futuras"] == 1 and rep["duplicadas"] == 1,
        str(rep))
    chk("parse: orden cronológico y sin duplicados", df["time"].is_monotonic_increasing and df["time"].is_unique)
    chk("parse: detecta lista desordenada", rep["desordenadas"])
    chk("parse: timestamps en UTC", str(df["time"].dt.tz) == "UTC")

    # 2) velas cerradas vs en formación y antigüedad
    closed, forming = split_closed(df, now)
    chk("cerradas: la vela de las 12:00:00 sigue formándose a las 12:00:30",
        len(forming) == 1 and forming["time"].iloc[0] == pd.Timestamp("2026-10-23 12:00:00", tz="UTC"))
    chk("cerradas: última cerrada = 11:59 y antigüedad 30 s",
        closed["time"].iloc[-1] == pd.Timestamp("2026-10-23 11:59:00", tz="UTC") and abs(candle_age_s(closed, now) - 30) < 1e-6,
        f"edad={candle_age_s(closed, now)}")
    chk("cerradas: sin datos → antigüedad infinita", candle_age_s(closed.iloc[0:0], now) == float("inf"))

    # 3) retraso detectado (simula una fuente atrasada 5 min)
    stale = parse_candles(_synth_raw(120, now.replace(second=0) - timedelta(minutes=6)), now)[0]
    age_stale = candle_age_s(split_closed(stale, now)[0], now)
    chk("retraso: una fuente atrasada supera el límite y se marca ROJO",
        age_stale > cfg.candle_max_s and freshness(1, 1, age_stale, 1, [], cfg)[0] == "red", f"edad={age_stale:.0f}s")

    # 4) huecos: internos se rellenan, el final NUNCA
    g = pd.DataFrame({"time": pd.to_datetime(["2026-10-23 11:50", "2026-10-23 11:52", "2026-10-23 11:53"], utc=True),
                      "open": [1, 1, 1], "high": [2, 2, 2], "low": [1, 1, 1], "close": [1.5, 1.5, 1.5], "volume": [1, 1, 1]})
    gf, nmiss = fill_gaps(g)
    chk("huecos: relleno interno de 1 min y gap_stats coincide", nmiss == 1 and len(gf) == 4 and gap_stats(g) == (1, 1))
    chk("huecos: nunca se extiende hacia el presente", gf["time"].iloc[-1] == g["time"].iloc[-1])

    # 5) fuente de respaldo: conversión del formato
    conv = _advanced_to_raw({"candles": [{"start": "1700000040", "low": "1", "high": "3", "open": "2", "close": "2.5", "volume": "4"}]})
    chk("respaldo: conversión Advanced → formato Exchange", conv == [[1700000040.0, 1.0, 3.0, 2.0, 2.5, 4.0]])
    try:
        _advanced_to_raw({"candles": []}); ok = False
    except RuntimeError:
        ok = True
    chk("respaldo: respuesta vacía lanza error (no se inventan velas)", ok)

    # 6) indicadores causales: añadir velas futuras no cambia los valores pasados
    big = parse_candles(_synth_raw(260, now.replace(second=0) - timedelta(minutes=1)), now)[0]
    f_full, f_cut = compute_features(big), compute_features(big.iloc[:200])
    cols = ["ema9", "ema21", "ema50", "rsi", "macd", "macd_hist", "atr", "sigma", "score"]
    chk("indicadores: causales (sin mirar el futuro)",
        all(np.allclose(f_full[c].iloc[199], f_cut[c].iloc[199], equal_nan=True) for c in cols))
    chk("indicadores: válidos tras el calentamiento", bool(f_full["valid"].iloc[-1]) and not f_full["valid"].iloc[10])

    # 7) Kalshi: precios derivados y coherencia
    m = parse_market({"ticker": "KXBTC15M-TEST", "yes_bid_dollars": "0.40", "no_bid_dollars": "0.55",
                      "floor_strike": 100000, "strike_type": "greater_or_equal",
                      "close_time": "2026-10-23T12:15:00Z", "yes_ask_size_fp": "120.00"})
    chk("kalshi: ask derivado = 1 − bid contrario", abs(m["yes_ask"] - 0.45) < 1e-9 and abs(m["no_ask"] - 0.60) < 1e-9
        and set(m["derived"]) == {"yes_ask", "no_ask"})
    chk("kalshi: precios coherentes aceptados / incoherentes rechazados",
        quote_check(m, 0.02)[0] == "ok" and quote_check({"yes_ask": 0.5, "no_bid": 0.2, "yes_bid": None, "no_ask": None}, 0.02)[0] == "fail")

    # 8) comisión y decisión
    chk("comisión taker: 0.07·0.5·0.5 = 1.75¢ y orden redondeada al centavo",
        abs(kalshi_fee(0.5, 0.07) - 0.0175) < 1e-12 and abs(kalshi_fee_order(10, 0.5, 0.07) - 0.18) < 1e-9)
    q_hi = {"yes_ask": 0.92, "yes_bid": 0.90, "no_ask": 0.10, "no_bid": 0.08, "yes_ask_size": 200, "yes_bid_size": 200}
    d1 = decide(0.90, q_hi, 5, cfg, True, [], "Baja")
    chk("decisión: 90 % de probabilidad a 92¢ NO es ventaja → ESPERAR", d1["signal"] == "ESPERAR")
    q_ok = {"yes_ask": 0.60, "yes_bid": 0.58, "no_ask": 0.42, "no_bid": 0.40, "yes_ask_size": 200, "yes_bid_size": 200}
    d2 = decide(0.80, q_ok, 5, cfg, True, [], "Baja")
    chk("decisión: 80 % a 60¢ con filtros OK → BUY UP (sin esperar al 80–90 % del mercado)", d2["signal"] == "BUY UP")
    d3 = decide(0.80, q_ok, 5, cfg, True, ["Velas atrasadas"], "Baja")
    chk("decisión: datos obsoletos bloquean la señal aunque haya ventaja", d3["signal"] == "ESPERAR")
    d4 = decide(0.80, dict(q_ok, yes_ask_size=1), 5, cfg, True, [], "Baja")
    chk("decisión: poca liquidez bloquea", d4["signal"] == "ESPERAR" and any(f["id"] == "depth" and f["status"] == "fail" for f in d4["filters"]))
    d5 = decide(0.20, q_ok, 5, cfg, True, [], "Baja")
    chk("decisión: probabilidad baja de YES con NO barato → BUY DOWN", d5["signal"] == "BUY DOWN")

    # 8b) inclinación (lean) y perfiles
    d6 = decide(0.62, q_ok, 5, cfg, True, [], "Baja")          # leve ventaja: no alcanza → ESPERAR con inclinación
    ln6 = compute_lean(0.62, d6)
    chk("inclinación: ESPERAR con p=62 % a 60¢ apunta a UP y muestra cuánto falta",
        d6["signal"] == "ESPERAR" and ln6 is not None and ln6["side"] == "UP" and ln6["missing"] > 0, str(ln6))
    chk("inclinación: no aparece cuando hay BUY", compute_lean(0.80, d2) is None)
    chk("inclinación: sin probabilidad no hay inclinación", compute_lean(None, d6) is None)
    q_mid = {"yes_ask": 0.50, "yes_bid": 0.48, "no_ask": 0.52, "no_bid": 0.50, "yes_ask_size": 200, "yes_bid_size": 200}
    ln7 = compute_lean(0.45, decide(0.45, q_mid, 5, cfg, True, [], "Baja"))
    chk("inclinación: p(YES)=45 % con ventaja débil apunta a DOWN", ln7 is not None and ln7["side"] == "DOWN", str(ln7))
    chk("perfiles: Sensible exige menos ventaja que Conservador",
        PROFILES["Sensible"][0] < PROFILES["Equilibrado"][0] < PROFILES["Conservador"][0])
    cfg_s = Cfg(min_edge=PROFILES["Sensible"][0] / 100, uncal_extra=PROFILES["Sensible"][1] / 100,
                slippage=PROFILES["Sensible"][2] / 100)
    chk("perfiles: una señal débil que Conservador rechaza, Sensible la acepta",
        decide(0.66, q_ok, 5, Cfg(), False, [], "Baja")["signal"] == "ESPERAR"
        and decide(0.66, q_ok, 5, cfg_s, False, [], "Baja")["signal"] == "BUY UP")

    # 9) modelo
    p_far = model_prob(100_300, 100_000, 5, 4e-4, 0.0, cfg)["p"]
    p_near = model_prob(100_010, 100_000, 5, 4e-4, 0.0, cfg)["p"]
    chk("modelo: más lejos sobre el umbral → más probabilidad de YES", p_far > p_near > 0.5)

    # 10) semáforo
    chk("semáforo: verde / amarillo / rojo",
        freshness(2, 3, 30, 2, [], cfg)[0] == "green" and freshness(2, 3, 100, 2, [], cfg)[0] == "yellow"
        and freshness(2, 3, 200, 2, [], cfg)[0] == "red" and freshness(None, 3, 30, 2, [], cfg)[0] == "red")

    # 10b) señal temprana: una por ventana, fija, sin exigir 80–90 %
    class _SS(dict):
        __getattr__ = dict.__getitem__
    _orig_save = globals()["save_history"]
    globals()["save_history"] = lambda h: None
    try:
        ws = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
        ctr_e = Contract("KXBTC15M-E1", "t", ws + timedelta(minutes=15), ws, 100_000.0, "umbral", True, True, [], "")
        row_e = pd.Series({"valid": True, "sigma": 4e-4, "score": 0.6})
        q_e = {"yes_ask": 0.55, "no_ask": 0.47}
        ss_e = _SS(history=[])
        e0 = compute_early(ss_e, cfg, ctr_e, ws + timedelta(seconds=10), row_e, 100_020.0, 14.8, q_e, "green")
        chk("temprana: antes del tiempo configurado queda pendiente", e0 is not None and e0["state"] == "pending")
        e1 = compute_early(ss_e, cfg, ctr_e, ws + timedelta(seconds=40), row_e, 100_020.0, 14.3, q_e, "green")
        chk("temprana: emite UP con tendencia alcista sin exigir 80–90 %",
            e1["state"] == "emitted" and e1["side"] == "UP" and e1["p"] < 0.80, str(e1))
        e2 = compute_early(ss_e, cfg, ctr_e, ws + timedelta(seconds=100), row_e.copy(), 99_900.0, 13.3, q_e, "green")
        chk("temprana: una sola por contrato y la dirección queda fija",
            len([h for h in ss_e["history"] if str(h["señal"]).startswith("EARLY")]) == 1 and e2["side"] == "UP"
            and e2["reversal"] is True, str(e2))
        e3 = compute_early(_SS(history=[]), cfg, ctr_e, ws + timedelta(seconds=40), row_e, 100_020.0, 14.3, q_e, "red")
        chk("temprana: con semáforo ROJO no se emite", e3["state"] == "blocked")
        chk("temprana: desactivada → None",
            compute_early(_SS(history=[]), replace(cfg, early_on=False), ctr_e, ws + timedelta(seconds=40), row_e,
                          100_020.0, 14.3, q_e, "green") is None)
    finally:
        globals()["save_history"] = _orig_save

    # 11) operación: cuerpo de la orden, guardas y firma
    chk("orden: BUY UP → yes + precio YES en dólares + fill_or_kill",
        build_order_body("KXBTC15M-X1", "UP", 2, 0.57, "b-X")
        == {"ticker": "KXBTC15M-X1", "side": "yes", "action": "buy", "client_order_id": "b-X", "count": 2,
            "count_fp": "2.00", "yes_price_dollars": "0.5700", "time_in_force": "fill_or_kill"})
    bd = build_order_body("T", "DOWN", 1, 0.43, "b-T")
    chk("orden: BUY DOWN → no + precio NO", bd["side"] == "no" and bd["no_price_dollars"] == "0.4300" and "yes_price_dollars" not in bd)
    chk("orden: client_order_id solo [A-Za-z0-9-]", re.fullmatch(r"[A-Za-z0-9-]+", client_order_id_for("KXBTC15M-26OCT231215-15_x")) is not None)
    chk("orden: precio límite redondea hacia arriba al centavo y tope 0.99", ceil_cent(0.917) == 0.92 and ceil_cent(0.60) == 0.60 and ceil_cent(0.995) == 0.99)
    ctr_ = type("C", (), {"ticker": "KXBTC15M-T1"})()
    Rg = {"signal": d2["signal"], "dec": d2, "fresh": {"level": "green"}, "ctr": ctr_}
    nowg = datetime(2026, 10, 23, 12, 0, 30, tzinfo=timezone.utc)
    tc = TradeCfg(mode="live", contracts=1)
    w0, plan0 = trade_guard(Rg, cfg, tc, [], nowg, 2, None)
    chk("guarda: todo en regla → se puede enviar", w0 == [] and plan0["side"] == "UP", str(w0))
    chk("guarda: semáforo no verde bloquea", trade_guard(dict(Rg, fresh={"level": "yellow"}), cfg, tc, [], nowg, 2, None)[0] != [])
    chk("guarda: señal no confirmada (1 lectura) bloquea", trade_guard(Rg, cfg, tc, [], nowg, 1, None)[0] != [])
    chk("guarda: una sola orden por contrato", trade_guard(Rg, cfg, tc, [{"ticker": "KXBTC15M-T1", "modo": "live"}], nowg, 2, None)[0] != [])
    chk("guarda: simulación previa no bloquea el modo real", trade_guard(Rg, cfg, tc, [{"ticker": "KXBTC15M-T1", "modo": "dry"}], nowg, 2, None)[0] == [])
    chk("guarda: tope por operación", trade_guard(Rg, cfg, TradeCfg(mode="live", contracts=20, max_cost=5.0), [], nowg, 2, None)[0] != [])
    lossy = [{"ticker": "A", "modo": "live", "ts_utc": "2026-10-23 10:00:00", "cierre_utc": "2026-10-23 10:15:00", "pnl": "-11", "fill": "1", "resultado": "no"}]
    chk("guarda: pérdida diaria máxima detiene", trade_guard(Rg, cfg, tc, lossy, nowg, 2, None)[0] != [])
    chk("guarda: operación detenida bloquea", trade_guard(Rg, cfg, tc, [], nowg, 2, "duda")[0] != [])
    chk("guarda: poca liquidez bloquea", trade_guard(dict(Rg, dec=d4), cfg, tc, [], nowg, 2, None)[0] != [])
    chk("guarda: sin señal BUY no opera", trade_guard(dict(Rg, signal="ESPERAR"), cfg, tc, [], nowg, 9, None)[0] != [])
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        sig = base64.b64decode(kalshi_sign("POST", "/trade-api/v2/portfolio/orders", "1700000000000", pem))
        k.public_key().verify(sig, b"1700000000000POST/trade-api/v2/portfolio/orders",
                              padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        chk("firma RSA-PSS verificable con la clave pública (ts+MÉTODO+ruta)", True)
    except ImportError:
        chk("firma RSA-PSS (omitida: falta 'cryptography' en este equipo)", True)
    except Exception as e:  # noqa: BLE001
        chk("firma RSA-PSS verificable", False, str(e))
    return T


def render_selftest() -> None:
    res = selftest()
    ok = sum(1 for _, c, _ in res if c)
    (st.success if ok == len(res) else st.error)(f"{ok}/{len(res)} pruebas superadas (datos sintéticos, sin internet).")
    st.markdown("".join(f"<div class='frow {'ok' if c else 'fail'}'><div class='fh'>{'✅' if c else '⛔'} {esc(n)}</div>"
                        + (f"<div class='fd'>{esc(d)}</div>" if d and not c else "") + "</div>" for n, c, d in res),
                unsafe_allow_html=True)


def live_check() -> None:
    """Consulta real (requiere internet). Imprime la antigüedad de la última vela cerrada por fuente y las
    cabeceras de caché: así se identifica de dónde viene el retraso en TU red."""
    print(f"Hora local del equipo (UTC): {utcnow():%Y-%m-%d %H:%M:%S}")
    for name, fn in CANDLE_SOURCES:
        try:
            t0 = utcnow()
            j, lat, sdt, meta = fn(t0)
            t1 = utcnow()
            df, rep = parse_candles(j, t1)
            closed, forming = split_closed(df, t1)
            print(f"\n[{name}] {lat:.0f} ms · intentos={meta['attempts']}")
            print(f"  última vela cerrada (inicio UTC): {closed['time'].iloc[-1]}  → antigüedad {candle_age_s(closed, t1):.0f} s")
            print(f"  vela en formación: {'sí' if len(forming) else 'no'} · filas={rep['filas']} · reporte={rep}")
            print(f"  cabecera Date: {sdt} (edad de la respuesta {(t1 - sdt).total_seconds():.1f} s)" if sdt else "  sin cabecera Date")
            print(f"  cabecera Age: {meta['age']} · caché: {meta['cache']}")
            print(f"  huecos (total, últimos 10): {gap_stats(closed)}")
        except Exception as e:  # noqa: BLE001
            print(f"\n[{name}] ERROR: {e}")
    try:
        tk = fetch_ticker()
        print(f"\n[Ticker] {tk['price']:,.2f} · última operación {tk['exchange_time']} · {tk['latency_ms']:.0f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"\n[Ticker] ERROR: {e}")
    try:
        k = fetch_kalshi_markets()
        print(f"[Kalshi] {len(k['markets'])} contratos abiertos · {k['latency_ms']:.0f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"[Kalshi] ERROR: {e}")


def render_help() -> None:
    st.markdown("""
### Cómo interpretar la señal
* **BUY UP / BUY DOWN** — El modelo estima una probabilidad mayor que el costo de equilibrio del lado (precio de compra +
  comisión + deslizamiento) más la ventaja mínima, y **todos** los filtros bloqueantes están en verde.
* **ESPERAR** — Falta algún dato fresco, hay contradicciones o la ventaja neta no alcanza. Abre **🔎 Diagnóstico**.
* **🧭 Inclinación** — Aparece bajo ESPERAR: hacia qué lado apunta el modelo y **cuántos ¢ de ventaja faltan**.
  Es solo informativa; **no es una señal de compra** y nunca dispara órdenes.
* Es una **estimación estadística**: nunca una certeza ni una garantía de ganancia.

### Señal temprana (una por ventana)
* Al inicio de cada ventana de 15 min la app emite **una dirección (UP/DOWN)** usando sobre todo la tendencia, **sin
  esperar** a que la probabilidad llegue a 80–90 %. Queda fijada toda la ventana; si el modelo cambia, solo avisa de un giro.
* Muestra el precio de compra y la **ventaja neta** real: si es negativa, el mercado ya cobra esa probabilidad.
* **No es la señal BUY** y nunca envía órdenes. Su acierto se mide en el historial (señales «EARLY»).

### Perfiles de sensibilidad
* **Conservador** (4¢ de ventaja, +2¢ si no calibrado, 1¢ de deslizamiento) · **Equilibrado** (2.5¢ / 1¢ / 0.5¢) ·
  **Sensible** (1.5¢ / 0¢ / 0.5¢).
* Un perfil más sensible da **más señales**, pero **no crea ventaja**: con comisiones, la rentabilidad esperada puede
  ser cercana a cero o negativa. Pruébalo primero en **Simulación**.

### Semáforo de frescura
* 🟢 Todo dentro de límites · 🟡 retraso leve o un intento fallido (se usa el dato anterior, con su edad real) ·
  🔴 alguna fuente crítica obsoleta o sin respuesta → las señales se bloquean.
* **Pantalla refrescada** (cada N s) ≠ **datos nuevos**: la app solo descarga de nuevo cuando pasó el intervalo mínimo
  de cada fuente, y la antigüedad siempre se cuenta desde la descarga original, nunca desde el redibujado.

### Sobre el retraso de las velas
Esta versión pide ventanas explícitas (`start`/`end`), envía `Cache-Control: no-cache`, mide la edad real de cada
respuesta (`Date`/`Age`), consulta una fuente de respaldo cuando la principal se atrasa y **bloquea** si ninguna entrega
datos recientes. No fabrica velas: los huecos al final se muestran como retraso. Ejecuta `python app.py --live-check`
en tu red para ver el retraso real de cada fuente.

### Operación automática (opcional)
Modos: **Apagado**, **Simulación** (registra lo que haría, no envía nada) y **REAL**. REAL exige `KALSHI_LIVE_OK=1`,
escribir OPERAR, semáforo verde, la misma señal en varias lecturas, una sola orden por contrato, tope por orden,
límite diario de operaciones, pérdida diaria máxima y exposición máxima. Las órdenes son límite *fill-or-kill*.
Si una orden queda en duda (timeout o error 5xx) todo se detiene hasta que revises tu cuenta. Empieza siempre en
Simulación y luego con 1 contrato. Una señal no es una garantía.

### Qué se bloquea
Datos frescos (precio, última operación, velas, huecos, Kalshi, reloj) · contrato correcto y reglas soportadas ·
referencia coherente · precios YES/NO coherentes · tiempo restante · ventaja neta · rango de entrada · spread ·
liquidez · incertidumbre. Los filtros no se relajan solos.

### Limitaciones importantes
* Kalshi liquida con **CF Benchmarks BRTI** (promedio de ~60 s). Este analizador usa **Coinbase** como aproximación.
* La comisión usa la fórmula general taker `0.07·P·(1−P)` (configurable). Verifica la tabla vigente de Kalshi.
* El score de tendencia/momentum es heurístico; hasta que el backtest lo respalde, la probabilidad se marca **no calibrada**.
* Los mercados de 15 minutos son muy eficientes y las comisiones consumen gran parte de cualquier ventaja pequeña.
  **Esto no es asesoría financiera. Puedes perder todo lo que operes.**
* Por defecto este programa **no envía órdenes** y no usa claves. La operación automática es opcional, está apagada
  y nunca guarda claves en el código ni en disco (solo variables de entorno).
""")
    st.markdown("#### Pruebas de diagnóstico")
    st.caption("Comprueban la lógica (timestamps, velas cerradas, huecos, caché, decisión) con datos SINTÉTICOS. "
               "No prueban tu conexión: para eso usa `python app.py --live-check`.")
    if st.button("Ejecutar pruebas"):
        render_selftest()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Kalshi BTC 15m · Analizador", page_icon="🟣", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    st.markdown("### 🟣 Kalshi · BTC 15 min")
    st.caption("Solo BTC/USD · la operación automática está APAGADA por defecto. Ajustes: botón » arriba a la izquierda.")

    global LOCAL_TZ
    with st.sidebar:
        st.header("Configuración")
        tzname = st.selectbox("Zona horaria", TZ_CHOICES,
                              index=TZ_CHOICES.index(DEFAULT_TZ) if DEFAULT_TZ in TZ_CHOICES else 0)
        LOCAL_TZ = ZoneInfo(tzname)
        refresh = st.slider("Redibujar pantalla cada (s)", 5, 30, 6)
        forced = None
        kal_side, _ = safe_fetch("Kalshi · mercados", fetch_kalshi_markets, 4)
        if kal_side is not None:
            tickers = [m["ticker"] for m in sorted(kal_side["markets"], key=lambda m: m["close_time"] or utcnow()) if m.get("ticker")]
            choice = st.selectbox("Contrato", ["Automático (el que está corriendo)"] + tickers)
            forced = None if choice.startswith("Automático") else choice
        else:
            st.caption("No se pudo listar contratos de Kalshi ahora mismo.")
        perfil = st.selectbox("Perfil de sensibilidad", list(PROFILES.keys()), index=1,
                              help="Más sensible = más señales, pero NO crea ventaja: con comisiones la "
                                   "rentabilidad esperada puede ser cercana a cero o negativa. Prueba en Simulación.")
        P_EDGE, P_EXTRA, P_SLIP = PROFILES[perfil]
        with st.expander("Umbrales de la señal (ajuste fino)", expanded=False):
            min_edge = st.slider("Ventaja neta mínima (¢)", 1.0, 15.0, P_EDGE, 0.5, key=f"edge_{perfil}")
            extra = st.slider("Margen extra si no está calibrado (¢)", 0.0, 10.0, P_EXTRA, 0.5, key=f"extra_{perfil}")
            max_spread = st.slider("Spread máximo (¢)", 1, 20, 8)
            min_depth = st.slider("Contratos mínimos al precio de entrada", 1, 100, 5)
            slip = st.slider("Deslizamiento supuesto (¢)", 0.0, 5.0, P_SLIP, 0.5, key=f"slip_{perfil}")
            fee_rate = st.number_input("Tasa de comisión taker", 0.0, 0.2, 0.07, 0.005, format="%.3f")
            min_tau = st.slider("Segundos mínimos restantes", 15, 180, 45)
        with st.expander("Frescura de datos", expanded=False):
            max_age = st.slider("Antigüedad máxima de precio/Kalshi (s)", 10, 60, 20)
            max_trade = st.slider("Antigüedad máxima de la última operación (s)", 20, 180, 60)
            c_warn = st.slider("Vela cerrada: aviso amarillo desde (s)", 60, 150, 75)
            c_max = st.slider("Vela cerrada: bloqueo desde (s)", 90, 300, 150)
            st.caption("Una vela de 1 min recién cerrada tiene entre 0 y ~60 s de antigüedad. Subir estos límites "
                       "permite señales con datos más viejos: hazlo solo si entiendes el riesgo.")
        with st.expander("Modelo", expanded=False):
            basis = st.slider("Error Coinbase↔BRTI (puntos base)", 0.0, 6.0, 1.5, 0.5)
            drift = st.slider("Peso máximo del score (σ)", 0.0, 0.6, 0.25, 0.05)
            infl = st.slider("Inflación de volatilidad", 1.0, 1.5, 1.10, 0.05)
            use_bt = st.checkbox("Aplicar resultados del último backtest", True)
        with st.expander("⚡ Señal temprana", expanded=False):
            early_on = st.checkbox("Emitir 1 señal por ventana de 15 min", True)
            early_after = st.slider("Emitirla a los (s) del inicio", 10, 180, 30, 5)
            early_lock = st.checkbox("Mantener la dirección toda la ventana", True)
            early_drift = st.slider("Peso de la tendencia (σ)", 0.2, 1.0, 0.6, 0.05)
        chart_n = st.slider("Velas en la gráfica", 30, 240, 60, 10)
        st.divider()
        with st.expander("🤖 Operación automática", expanded=False):
            st.caption("Apagada por defecto. Claves solo por variables de entorno (KALSHI_API_KEY_ID, "
                       "KALSHI_PRIVATE_KEY_PATH). Empieza en Simulación.")
            mode_l = st.radio("Modo", ["Apagado", "Simulación (no envía nada)", "REAL (envía órdenes)"], index=0)
            mode = {"Apagado": "off", "Simulación (no envía nada)": "dry", "REAL (envía órdenes)": "live"}[mode_l]
            t_contracts = st.slider("Contratos por orden", 1, 20, 1)
            t_max_cost = st.number_input("Tope por operación ($)", 0.5, 100.0, 5.0, 0.5)
            t_max_trades = st.slider("Operaciones reales máx. por día", 1, 20, 5)
            t_max_loss = st.number_input("Pérdida diaria máxima ($)", 1.0, 500.0, 10.0, 1.0)
            t_max_exp = st.number_input("Exposición máxima sin resolver ($)", 1.0, 500.0, 10.0, 1.0)
            t_confirm = st.slider("Lecturas seguidas con la misma señal", 1, 5, 2)
            unlocked = False
            if mode == "live":
                st.warning("REAL envía órdenes con dinero de verdad. Requiere KALSHI_LIVE_OK=1 en el entorno.")
                typed = st.text_input("Escribe OPERAR para confirmar")
                unlocked = (os.getenv("KALSHI_LIVE_OK") == "1") and typed.strip() == "OPERAR"
        tcfg = TradeCfg(mode=mode, contracts=t_contracts, max_cost=t_max_cost, max_trades_day=t_max_trades,
                        max_daily_loss=t_max_loss, max_exposure=t_max_exp, confirm_reads=t_confirm, live_unlocked=unlocked)
        st.caption("Datos públicos sin claves. Las claves solo se usan si activas la operación automática.")

    cfg = Cfg(refresh_s=refresh, min_edge=min_edge / 100, uncal_extra=extra / 100, max_spread=max_spread / 100,
              min_depth=float(min_depth), fee_rate=fee_rate, slippage=slip / 100, min_tau_min=min_tau / 60,
              basis_bps=basis, drift_cap=drift, sigma_inflate=infl, chart_minutes=chart_n, use_backtest=use_bt,
              max_age_s=max_age, max_trade_age_s=max_trade, candle_warn_s=min(c_warn, c_max), candle_max_s=c_max,
              early_on=early_on, early_after_s=early_after, early_lock=early_lock, early_drift=early_drift)

    t_live, t_bt, t_help = st.tabs(["Señal", "Backtest", "Ayuda"])
    with t_live:
        live = st.fragment(run_every=cfg.refresh_s)(render_live)
        live(cfg, forced, tcfg)
    with t_bt:
        render_backtest(cfg)
    with t_help:
        render_help()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _res = selftest()
        for _n, _c, _d in _res:
            print(("PASS " if _c else "FAIL ") + _n + (f"  [{_d}]" if (_d and not _c) else ""))
        print(f"\n{sum(1 for _, c, _ in _res if c)}/{len(_res)} pruebas superadas (datos sintéticos)")
        sys.exit(0 if all(c for _, c, _ in _res) else 1)
    elif "--live-check" in sys.argv:
        live_check()
        sys.exit(0)
    else:
        main()
