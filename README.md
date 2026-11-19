Kalshi BTC 15m — bot de señales (PAPER)

Bot informativo para contratos BTC de 15 minutos de Kalshi. Genera `STRONG BUY UP`, `BUY UP`, `STRONG BUY DOWN`, `BUY DOWN` o `WAIT`. **No envía órdenes reales.** No garantiza un win rate concreto.

## Instalación (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

Consola, sin dashboard:

```powershell
python run_bot.py
```

## Cómo calcula las señales

- Descarga velas cerradas de 1 minuto de Coinbase y el mercado abierto de la serie `KXBTC15M`.
- Calcula RSI, EMA 9/21/50, MACD, ATR, bandas de Bollinger, retornos de 1/5/15 minutos y volatilidad realizada.
- Estima una probabilidad base a partir de distancia precio/strike, tiempo restante y volatilidad de un minuto; incorpora un drift de corto plazo limitado.
- Aplica un ajuste técnico pequeño y acotado por consenso de indicadores.
- Calcula por separado el edge estimado de UP y DOWN después de comisión y slippage configurados.
- Rechaza datos incompletos, contratos cerca del cierre, precios fuera de rango y spreads demasiado amplios.

### Corrección importante de volatilidad

La volatilidad de velas de 1 minuto se mantiene como desviación estándar de retornos por raíz de minuto. No se multiplica por `sqrt(60)`. La versión anterior hacía esa conversión y luego volvía a escalar por `sqrt(minutos_restantes)`, lo que podía aplanar demasiado las probabilidades.

## Por qué puede mostrar WAIT

`WAIT` no es un error por sí solo. Aparece cuando no hay datos fiables, falta una cotización necesaria, el spread es excesivo, el contrato está por cerrar, la confianza direccional no llega al mínimo o el edge neto estimado no supera los costes y umbrales configurados. El panel debe mostrar el motivo. **No se fuerza una compra para evitar WAIT**, porque eso aumentaría entradas sin ventaja demostrada.

## Configuración

Edita `config.toml`. `min_edge` y `strong_edge` representan ventaja neta estimada por contrato, no porcentaje de aciertos. `fee_rate` y `slippage` son supuestos que debes contrastar con las tarifas y ejecución reales de Kalshi.

## Limitaciones importantes

- Las probabilidades son estimaciones heurísticas; no están calibradas ni validadas para prometer 90% de aciertos.
- La disponibilidad de campos de strike y cotizaciones depende de la respuesta de la API de Kalshi. Si no se reciben, el bot bloquea la señal.
- Esta reconstrucción registra señales, pero todavía no implementa una liquidación shadow completa contra el resultado oficial ni un backtest con histórico real de precios de contratos. Por tanto, las métricas de rentabilidad no deben interpretarse como resultados de trading.
- El precio de Coinbase no garantiza que se pueda ejecutar al mismo precio en Kalshi. Spread, profundidad, latencia y comisiones reales pueden cambiar el resultado.
- Antes de considerar dinero real, recopila cientos de señales con datos completos de mercado, guarda bid/ask al momento de entrada y resultado oficial al vencimiento, y valida EV neto fuera de muestra.

## Estructura

- `app.py`: dashboard Streamlit.
- `run_bot.py`: salida en consola.
- `btcbot/market_data.py`: datos de Coinbase y Kalshi.
- `btcbot/features.py`: indicadores causales sobre velas cerradas.
- `btcbot/signal_engine.py`: probabilidades, edge y clasificación de señal.
- `btcbot/probability.py`: distribución normal.
- `btcbot/tracking.py`: registro local de señales JSONL.
- `tests/`: pruebas unitarias básicas.
