from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import pandas as pd
from app.analysis_engine import compute_indicators, decide_signal

app = FastAPI(title="Trading Signal Service", version="2.0.0")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/signal")
def signal(
    pair: str = Query("EURUSD=X"),
    interval: str = Query("1h"),
    period: str = Query("60d"),
):
    try:
        df: pd.DataFrame = yf.download(pair, interval=interval, period=period, progress=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"yfinance error: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data returned for given parameters")

    cols = {c.lower(): c for c in df.columns}
    try:
        data = df.rename(columns={
            cols.get('open', 'Open'): 'Open',
            cols.get('high', 'High'): 'High',
            cols.get('low', 'Low'): 'Low',
            cols.get('close', 'Close'): 'Close',
            cols.get('volume', 'Volume'): 'Volume',
        })[['Open','High','Low','Close','Volume']].copy()
    except Exception:
        try:
            data = df[['Open','High','Low','Close','Volume']].copy()
        except Exception:
            raise HTTPException(status_code=500, detail=f"Unexpected columns: {list(df.columns)}")

    ind = compute_indicators(data).dropna()
    if ind.empty:
        raise HTTPException(status_code=422, detail="Not enough data after indicators")

    latest = ind.iloc[-1]
    sig, reasons = decide_signal(latest)

    return {
        "pair": pair,
        "interval": interval,
        "period": period,
        "signal": sig,
        "close": float(latest["Close"]),
        "rsi_14": float(latest["rsi_14"]),
        "ema_fast": float(latest["ema_fast"]),
        "ema_slow": float(latest["ema_slow"]),
        "macd": float(latest["macd"]),
        "macd_signal": float(latest["macd_signal"]),
        "macd_hist": float(latest["macd_hist"]),
        "bb_lower": float(latest["bb_lower"]),
        "bb_mid": float(latest["bb_mid"]),
        "bb_upper": float(latest["bb_upper"]),
        "atr_14": float(latest["atr_14"]),
        "reasons": reasons,
    }
