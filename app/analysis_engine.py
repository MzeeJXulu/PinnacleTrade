# app/analysis_engine.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Indicators from 'ta' (works with pandas 2.2.x and Py3.11/3.12)
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def compute_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_len: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """
    df index: Datetime; columns: ['Open','High','Low','Close','Volume'].
    Returns a DataFrame with:
      ema_fast, ema_slow, rsi_{len}, macd, macd_signal, macd_hist, bb_lower, bb_mid, bb_upper, atr_14
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Trend
    ema_f = EMAIndicator(close=data["Close"], window=ema_fast)
    ema_s = EMAIndicator(close=data["Close"], window=ema_slow)
    data["ema_fast"] = ema_f.ema_indicator()
    data["ema_slow"] = ema_s.ema_indicator()

    # Momentum
    rsi = RSIIndicator(close=data["Close"], window=rsi_len)
    data[f"rsi_{rsi_len}"] = rsi.rsi()

    macd = MACD(
        close=data["Close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()

    # Volatility
    bb = BollingerBands(close=data["Close"], window=20, window_dev=2.0)
    data["bb_lower"] = bb.bollinger_lband()
    data["bb_mid"] = bb.bollinger_mavg()
    data["bb_upper"] = bb.bollinger_hband()

    atr = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["atr_14"] = atr.average_true_range()

    # Clean
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    return data


def decide_signal(
    latest_row: pd.Series,
    rsi_len: int = 14,
    rsi_buy: float = 30.0,
    rsi_sell: float = 70.0,
    macd_hist_th: float = 0.0,
) -> tuple[str, list[str]]:
    """
    BUY if RSI <= rsi_buy AND ema_fast > ema_slow AND macd_hist >= +th
    SELL if RSI >= rsi_sell AND ema_fast < ema_slow AND macd_hist <= -th
    else HOLD
    """
    reasons: list[str] = []
    rsi_val = float(latest_row.get(f"rsi_{rsi_len}", np.nan))
    ema_fast_v = float(latest_row.get("ema_fast", np.nan))
    ema_slow_v = float(latest_row.get("ema_slow", np.nan))
    macd_hist_v = float(latest_row.get("macd_hist", np.nan))

    if any(np.isnan(v) for v in (rsi_val, ema_fast_v, ema_slow_v, macd_hist_v)):
        return "HOLD", ["Indicators not available"]

    buy = (rsi_val <= rsi_buy) and (ema_fast_v > ema_slow_v) and (macd_hist_v >= abs(macd_hist_th))
    sell = (rsi_val >= rsi_sell) and (ema_fast_v < ema_slow_v) and (macd_hist_v <= -abs(macd_hist_th))

    if buy:
        reasons.append(f"RSI({rsi_len}) ≤ {rsi_buy} ({rsi_val:.2f})")
        reasons.append(f"EMA fast({ema_fast_v:.6f}) > EMA slow({ema_slow_v:.6f})")
        reasons.append(f"MACD hist ≥ {abs(macd_hist_th)} ({macd_hist_v:.6f})")
        return "BUY", reasons

    if sell:
        reasons.append(f"RSI({rsi_len}) ≥ {rsi_sell} ({rsi_val:.2f})")
        reasons.append(f"EMA fast({ema_fast_v:.6f}) < EMA slow({ema_slow_v:.6f})")
        reasons.append(f"MACD hist ≤ -{abs(macd_hist_th)} ({macd_hist_v:.6f})")
        return "SELL", reasons

    return "HOLD", ["No strong multi-indicator agreement"]
