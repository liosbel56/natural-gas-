# Kalshi BTC 15-min — Bot de señales (entregas 1–5)

Genera señales `STRONG BUY UP / BUY UP / STRONG BUY DOWN / BUY DOWN / WAIT` para los contratos BTC de 15 minutos
de Kalshi (serie `KXBTC15M`) y las evalúa en **paper + shadow trading**. **No ejecuta operaciones reales.**
La configuración rechaza cualquier modo distinto de `paper`.

## Instalación y uso en Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # si falla: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python -m unittest discover -s tests      # deben pasar los 63 tests

python run_collector.py --cycles 5        # 1) prueba de conectividad real (debe decir [OK])
python download_history.py --days 60      # 2) historial de velas 1m de Coinbase (~5 min)
python train_and_backtest.py              # 3) entrena, valida walk-forward, backtest -> reports/report.md
streamlit run app.py                      # 4) dashboard (arranca el motor en vivo)
# alternativa sin dashboard:  python run_bot.py
```

## Cómo decide

1. **Probabilidad base** `P(UP) = Φ(z)`, con `z = ln(precio/strike) / (σ·√minutos_restantes)`. Es la pieza
   que más pesa: en un contrato "BTC ≥ strike al cierre", la distancia al strike frente a la volatilidad lo explica casi todo.
2. **ML opcional** (logreg, HistGradientBoosting, RandomForest, y XGBoost/LightGBM/CatBoost si los instalas), con
   calibración de Platt. Solo se usa si supera a Φ(z) **fuera de muestra** (lo decide `train_and_backtest.py`).
3. **Ventaja neta** = `p − precio_ask − comisión − slippage`. Solo hay señal si supera un umbral que se **adapta**
   (sube con el spread, picos de volatilidad, incertidumbre del modelo y order book en contra) pero nunca se baja
   para forzar operaciones. Si no hay ventaja, `WAIT` con el motivo exacto.
4. **Filtros**: no entra en los últimos 45 s, spread > 0.06, precios fuera de [0.10, 0.90], datos no confiables
   (→ `DATA UNAVAILABLE` y se bloquean señales).
5. **Riesgo**: Kelly fraccional (¼), máx. 2 % por operación, 5 % de exposición, límite de pérdida diaria 5 %, pausa
   tras 5 pérdidas seguidas.
6. **Shadow trading**: cada señal se registra y se liquida con el resultado oficial de Kalshi. Una entrada por contrato.

## Lo que este proyecto NO puede afirmar (léelo)

- **El 90 % de win rate no está demostrado ni es de esperar.** El win rate depende del precio al que compras:
  acertar 90 % comprando a 0.92 pierde dinero. La métrica que importa es el **EV neto por operación**.
- **El backtest usa un precio de mercado proxy** (Φ(z) ± spread) y un strike proxy (apertura de la vela de 1 min),
  porque no hay un histórico gratuito y completo de precios de contratos Kalshi. Mide si el modelo mejora a una
  fórmula ingenua, **no** la rentabilidad real. La única prueba real es el shadow trading con precios reales
  acumulados durante semanas (cientos de operaciones) y su intervalo de confianza.
- **Sin datos históricos de libro de órdenes**, el order book no entra al entrenamiento; solo actúa como filtro en vivo.
- **No incluí LSTM/TCN/Transformers**: con datos tabulares muy correlacionados rara vez superan a los árboles y
  exigen PyTorch. El marco walk-forward permite añadirlos después si lo pides.
- **Verifica la tarifa de comisión de Kalshi** (`signals.fee_rate`, por defecto 0.07): cambia directamente el edge neto.
- Pruebas hechas aquí: los 63 tests unitarios y un chequeo con datos sintéticos sin ventaja (el pipeline no inventa
  edge: el ML no supera a la base y todas las configuraciones pierden tras comisiones). **No pude probar contra las APIs
  reales ni ejecutar Streamlit** (sin internet ni Streamlit instalado en mi entorno).

## Estructura

| Archivo | Función |
|---|---|
| `config.toml` | Toda la configuración (fuentes, umbrales, riesgo) |
| `btcbot/exchanges.py`, `kalshi.py`, `http.py`, `collector.py` | Datos de Coinbase/Binance/Kraken/Kalshi con reintentos |
| `btcbot/health.py` | Reglas de `DATA UNAVAILABLE` |
| `btcbot/candles.py` | Velas 5s/15s/1m/5m solo cerradas |
| `btcbot/features.py` | RSI, MACD, EMAs, Bollinger, ATR, ADX, OBV, volatilidad realizada, z-score... y dataset |
| `btcbot/probability.py` | Φ(z), Monte Carlo bootstrap, comisiones |
| `btcbot/signal_engine.py` | Decisión, Confidence Score, riesgos de reversión/lateral/liquidez |
| `btcbot/ml.py` | Modelos, calibración, walk-forward con purga y embargo |
| `btcbot/backtest.py`, `metrics.py` | Win rate, IC Wilson, PF, ROI, EV, MDD, Sharpe, curva frecuencia-precisión |
| `btcbot/risk.py` | Cuenta paper, Kelly fraccional, límites |
| `btcbot/tracking.py` | Registro de señales, liquidación, estadísticas por hora/lado/régimen |
| `btcbot/engine.py`, `bootstrap.py` | Motor en vivo |
| `app.py` | Dashboard Streamlit |
| `train_and_backtest.py`, `download_history.py`, `run_bot.py`, `run_collector.py` | Scripts |

## Cómo se decide si el bot sirve

Deja correr `run_bot.py` o el dashboard varias semanas. Mira la pestaña **Rendimiento**: el bot solo merece capital real
si, con cientos de operaciones shadow, el **límite inferior del IC 95 %** del EV por operación es positivo y
el drawdown es tolerable. Ese es el resultado verificable que pediste; hasta entonces, la capacidad real es desconocida.
