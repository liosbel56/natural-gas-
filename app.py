
# =====================================================
# SEÑALES ESTRICTAS: 4 DE 4 CONFIRMACIONES
# =====================================================
def make_signal(df, seconds_left=None):
    row = df.iloc[-1]
    previous = df.iloc[-2]

    reasons = []
    score = 0

    # 1. Evitar señales cerca del cierre
    if seconds_left is not None and seconds_left < 30:
        return "ESPERAR", 0, [
            "Quedan menos de 30 segundos para el cierre.",
            "Entrada bloqueada por seguridad."
        ]

    # 2. Comprobar datos válidos
    required = [
        "ema9", "ema21", "ema50",
        "macd", "macd_signal",
        "momentum3", "rsi"
    ]

    for col in required:
        if col not in df.columns:
            return "ESPERAR", 0, [
                f"Falta el indicador {col}."
            ]

        if not np.isfinite(row[col]):
            return "ESPERAR", 0, [
                f"Datos insuficientes para {col}."
            ]

    # 3. Indicador 1: EMA 9 frente a EMA 21
    ema_up = row["ema9"] > row["ema21"]
    ema_down = row["ema9"] < row["ema21"]

    reasons.append(
        "EMA 9/21: " +
        ("ALCISTA" if ema_up else
         "BAJISTA" if ema_down else "NEUTRAL")
    )

    # 4. Indicador 2: tendencia EMA 21 frente a EMA 50
    trend_up = row["ema21"] > row["ema50"]
    trend_down = row["ema21"] < row["ema50"]

    reasons.append(
        "Tendencia EMA 21/50: " +
        ("ALCISTA" if trend_up else
         "BAJISTA" if trend_down else "NEUTRAL")
    )

    # 5. Indicador 3: MACD
    macd_up = row["macd"] > row["macd_signal"]
    macd_down = row["macd"] < row["macd_signal"]

    reasons.append(
        "MACD: " +
        ("ALCISTA" if macd_up else
         "BAJISTA" if macd_down else "NEUTRAL")
    )

    # 6. Indicador 4: Momentum de 3 minutos
    momentum = row["momentum3"]

    momentum_up = momentum > 0
    momentum_down = momentum < 0

    reasons.append(
        f"Momentum 3m: {momentum:+.4f}% " +
        ("ALCISTA" if momentum_up else
         "BAJISTA" if momentum_down else "NEUTRAL")
    )

    # 7. RSI como filtro adicional obligatorio
    rsi = float(row["rsi"])
    rsi_up = 50 < rsi < 70
    rsi_down = 30 < rsi < 50

    reasons.append(f"RSI: {rsi:.1f}")

    # 8. Detectar cruces recientes contrarios
    bearish_cross = (
        previous["ema9"] >= previous["ema21"]
        and row["ema9"] < row["ema21"]
    )

    bullish_cross = (
        previous["ema9"] <= previous["ema21"]
        and row["ema9"] > row["ema21"]
    )

    # 9. Calcular puntuación direccional: máximo +4 / mínimo -4
    score = (
        (1 if ema_up else -1 if ema_down else 0)
        + (1 if trend_up else -1 if trend_down else 0)
        + (1 if macd_up else -1 if macd_down else 0)
        + (1 if momentum_up else -1 if momentum_down else 0)
    )

    # 10. EXIGIR 4 DE 4, RSI válido y sin cruce contrario
    confirm_up = (
        ema_up
        and trend_up
        and macd_up
        and momentum_up
        and rsi_up
        and not bearish_cross
    )

    confirm_down = (
        ema_down
        and trend_down
        and macd_down
        and momentum_down
        and rsi_down
        and not bullish_cross
    )

    if confirm_up:
        signal = "COMPRAR UP"
        reasons.append(
            "CONFIRMADO: 4/4 indicadores alcistas y RSI válido."
        )

    elif confirm_down:
        signal = "COMPRAR DOWN"
        reasons.append(
            "CONFIRMADO: 4/4 indicadores bajistas y RSI válido."
        )

    else:
        signal = "ESPERAR"
        reasons.append(
            "No se cumplen todas las confirmaciones. "
            "No hay señal de compra."
        )

    return signal, score, reasons
