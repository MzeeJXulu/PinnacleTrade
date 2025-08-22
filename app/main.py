from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import yfinance as yf
import pandas as pd
from app.analysis_engine import generate_signal

app = FastAPI(title="Pandas-TA Signal Service", version="1.0.0")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/signal")
def signal(
    ticker: str = Query(..., description="Ticker symbol, e.g., SPY"),
    interval: str = Query("1d", description="yfinance interval, e.g., 1d, 1h, 5m"),
    period: str = Query("180d", description="yfinance period, e.g., 60d, 1y"),
):
    try:
        df: pd.DataFrame = yf.download(ticker, interval=interval, period=period, progress=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"yfinance error: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data returned for given parameters")

    # normalize columns (sometimes yfinance returns lowercase or multiindex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([c for c in tup if c]) for tup in df.columns]
    cols = {c.lower(): c for c in df.columns}
    # Map required columns
    try:
        data = df.rename(columns={
            cols.get('open', 'Open'): 'Open',
            cols.get('high', 'High'): 'High',
            cols.get('low', 'Low'): 'Low',
            cols.get('close', 'Close'): 'Close',
            cols.get('adj close', cols.get('adj_close', 'Adj Close')): 'Adj Close',
            cols.get('volume', 'Volume'): 'Volume',
        })[['Open','High','Low','Close','Volume']].copy()
    except Exception:
        # Fallback if columns already in expected case
        try:
            data = df[['Open','High','Low','Close','Volume']].copy()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected data format: {list(df.columns)}")

    result = generate_signal(data)
    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "period": period,
        **result,
    }

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
