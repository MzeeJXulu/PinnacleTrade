from __future__ import annotations
import pandas as pd
import pandas_ta as ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: ['Open','High','Low','Close','Volume']
    out = df.copy()

    # Basic indicators
    out['rsi'] = ta.rsi(out['Close'], length=14)
    out['ema_fast'] = ta.ema(out['Close'], length=9)
    out['ema_slow'] = ta.ema(out['Close'], length=21)

    # Simple signal logic
    out['ema_cross'] = (out['ema_fast'] > out['ema_slow']).astype(int)
    out['overbought'] = (out['rsi'] >= 70).astype(int)
    out['oversold'] = (out['rsi'] <= 30).astype(int)

    return out

def generate_signal(df: pd.DataFrame) -> dict:
    out = compute_indicators(df).dropna()
    if out.empty:
        return {
            "signal": "hold",
            "reason": "Not enough data after indicators",
        }

    last = out.iloc[-1]
    signal = "hold"
    reasons = []

    if last['ema_cross'] and last['oversold']:
        signal = "buy"
        reasons.append("EMA(9) > EMA(21) and RSI <= 30")
    elif (not last['ema_cross']) and last['overbought']:
        signal = "sell"
        reasons.append("EMA(9) < EMA(21) and RSI >= 70")
    else:
        reasons.append("No strong condition met")

    return {
        "signal": signal,
        "rsi": float(last['rsi']),
        "ema_fast": float(last['ema_fast']),
        "ema_slow": float(last['ema_slow']),
        "close": float(last['Close']),
        "reasons": reasons,
    }
