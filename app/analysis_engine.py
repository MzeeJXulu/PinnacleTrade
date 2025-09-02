
from __future__ import annotations
import pandas as pd
import pandas_ta as ta

def compute_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    bb_len: int = 20,
    bb_std: float = 2.0,
    rsi_len: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    data = df.copy()
    data[f"rsi_{rsi_len}"] = ta.rsi(data["Close"], length=rsi_len)
    data["ema_fast"] = ta.ema(data["Close"], length=ema_fast)
    data["ema_slow"] = ta.ema(data["Close"], length=ema_slow)
    macd = ta.macd(data["Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data["macd"] = macd[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
    data["macd_signal"] = macd[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
    data["macd_hist"] = macd[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]
    bb = ta.bbands(data["Close"], length=bb_len, std=bb_std)
    data["bb_lower"] = bb[f"BBL_{bb_len}_{bb_std}"]
    data["bb_mid"]   = bb[f"BBM_{bb_len}_{bb_std}"]
    data["bb_upper"] = bb[f"BBU_{bb_len}_{bb_std}"]
    data["atr_14"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)
    return data

def decide_signal(
    latest: pd.Series,
    rsi_len: int = 14,
    rsi_buy: float = 30.0,
    rsi_sell: float = 70.0,
    macd_hist_th: float = 0.0
):
    rsi_col = f"rsi_{rsi_len}"
    reasons = []
    buy = (latest[rsi_col] < rsi_buy) and (latest["ema_fast"] > latest["ema_slow"]) and (latest["macd_hist"] > macd_hist_th)
    sell = (latest[rsi_col] > rsi_sell) and (latest["ema_fast"] < latest["ema_slow"]) and (latest["macd_hist"] < -macd_hist_th)
    if buy:
        reasons.append(f"RSI<{rsi_buy}, EMA fast>slow, MACD hist>{macd_hist_th}")
        return "BUY", reasons
    if sell:
        reasons.append(f"RSI>{rsi_sell}, EMA fast<slow, MACD hist<-{macd_hist_th}")
        return "SELL", reasons
    if latest["Close"] <= latest["bb_lower"]:
        reasons.append("Price near/below lower Bollinger band (bounce?)")
    elif latest["Close"] >= latest["bb_upper"]:
        reasons.append("Price near/above upper Bollinger band (mean reversion?)")
    else:
        reasons.append("No strong multi-indicator agreement")
    return "HOLD", reasons
