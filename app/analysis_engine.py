from __future__ import annotations
import pandas as pd
import pandas_ta as ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["rsi_14"] = ta.rsi(data["Close"], length=14)
    data["ema_fast"] = ta.ema(data["Close"], length=12)
    data["ema_slow"] = ta.ema(data["Close"], length=26)
    macd = ta.macd(data["Close"], fast=12, slow=26, signal=9)
    data["macd"] = macd["MACD_12_26_9"]
    data["macd_signal"] = macd["MACDs_12_26_9"]
    data["macd_hist"] = macd["MACDh_12_26_9"]
    bb = ta.bbands(data["Close"], length=20, std=2.0)
    data["bb_lower"] = bb["BBL_20_2.0"]
    data["bb_mid"]   = bb["BBM_20_2.0"]
    data["bb_upper"] = bb["BBU_20_2.0"]
    data["atr_14"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)
    return data

def rule_buy(row) -> bool:
    return (row["rsi_14"] < 30) and (row["ema_fast"] > row["ema_slow"]) and (row["macd_hist"] > 0)

def rule_sell(row) -> bool:
    return (row["rsi_14"] > 70) and (row["ema_fast"] < row["ema_slow"]) and (row["macd_hist"] < 0)

def decide_signal(latest: pd.Series):
    reasons = []
    if rule_buy(latest):
        reasons.append("RSI<30, EMA fast>slow, MACD histogram>0")
        return "BUY", reasons
    if rule_sell(latest):
        reasons.append("RSI>70, EMA fast<slow, MACD histogram<0")
        return "SELL", reasons
    if latest["Close"] <= latest["bb_lower"]:
        reasons.append("Price near/below lower Bollinger band (bounce?)")
    elif latest["Close"] >= latest["bb_upper"]:
        reasons.append("Price near/above upper Bollinger band (mean reversion?)")
    else:
        reasons.append("No strong multi-indicator agreement")
    return "HOLD", reasons
