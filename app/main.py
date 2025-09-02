
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
import os, requests
import yfinance as yf
import pandas as pd
from app.analysis_engine import compute_indicators, decide_signal

app = FastAPI(title="Trading Signal Service + Dashboard + Webhook", version="3.0.0")

@app.get("/")
def root():
    return {"status": "ok"}

def sparkline_svg(series, width=720, height=160, pad=12):
    vals = [float(x) for x in series if pd.notna(x)]
    if not vals or len(vals) < 2:
        return f"<svg width='{width}' height='{height}'></svg>"
    vmin, vmax = min(vals), max(vals)
    vrange = (vmax - vmin) or 1.0
    step = (width - 2*pad) / (len(vals) - 1)
    pts = []
    for i, v in enumerate(vals):
        x = pad + i * step
        y = pad + (height - 2*pad) * (1 - (v - vmin) / vrange)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    last_x, last_y = pts[-1].split(",")
    last_y = float(last_y)
    color = "#16a34a" if vals[-1] >= vals[0] else "#dc2626"
    return f"""<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>
      <rect x='0' y='0' width='{width}' height='{height}' fill='#0b1020' rx='10'/>
      <polyline fill='none' stroke='{color}' stroke-width='2' points='{poly}'/>
      <circle cx='{last_x}' cy='{last_y:.1f}' r='3' fill='{color}' />
    </svg>"""

@app.get("/signal")
def signal(
    pair: str = Query("EURUSD=X"),
    interval: str = Query("1h"),
    period: str = Query("60d"),
    ema_fast: int = Query(12, ge=2, le=200),
    ema_slow: int = Query(26, ge=3, le=400),
    rsi_len: int = Query(14, ge=2, le=200),
    rsi_buy: float = Query(30.0, ge=0, le=100),
    rsi_sell: float = Query(70.0, ge=0, le=100),
    macd_fast: int = Query(12, ge=2, le=200),
    macd_slow: int = Query(26, ge=3, le=400),
    macd_signal: int = Query(9, ge=2, le=200),
    macd_hist_th: float = Query(0.0, ge=0, le=5.0),
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

    ind = compute_indicators(
        data,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_len=rsi_len,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    ).dropna()

    if ind.empty:
        raise HTTPException(status_code=422, detail="Not enough data after indicators")

    latest = ind.iloc[-1]
    sig, reasons = decide_signal(
        latest,
        rsi_len=rsi_len,
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell,
        macd_hist_th=macd_hist_th,
    )

    payload = {
        "pair": pair,
        "interval": interval,
        "period": period,
        "signal": sig,
        "close": float(latest["Close"]),
        f"rsi_{rsi_len}": float(latest[f"rsi_{rsi_len}"]),
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
        "params": {
            "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_len": rsi_len,
            "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
            "macd_fast": macd_fast, "macd_slow": macd_slow, "macd_signal": macd_signal,
            "macd_hist_th": macd_hist_th,
        }
    }

    url = os.getenv("WEBHOOK_URL", "").strip()
    if url:
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass

    return payload

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    pair: str = Query("EURUSD=X"),
    interval: str = Query("1h"),
    period: str = Query("60d"),
    ema_fast: int = Query(12),
    ema_slow: int = Query(26),
    rsi_len: int = Query(14),
    rsi_buy: float = Query(30.0),
    rsi_sell: float = Query(70.0),
    macd_fast: int = Query(12),
    macd_slow: int = Query(26),
    macd_signal: int = Query(9),
    macd_hist_th: float = Query(0.0),
):
    try:
        df: pd.DataFrame = yf.download(pair, interval=interval, period=period, progress=False)
    except Exception as e:
        return HTMLResponse(f"<pre>yfinance error: {e}</pre>", status_code=400)

    if df is None or df.empty:
        return HTMLResponse("<h2>No data returned</h2>", status_code=404)

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
            return HTMLResponse(f"<pre>Unexpected columns: {list(df.columns)}</pre>", status_code=500)

    ind = compute_indicators(
        data,
        ema_fast=ema_fast, ema_slow=ema_slow,
        rsi_len=rsi_len,
        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal
    ).dropna()

    if ind.empty:
        return HTMLResponse("<h2>Not enough data after indicators</h2>", status_code=422)

    latest = ind.iloc[-1]
    sig, reasons = decide_signal(latest, rsi_len=rsi_len, rsi_buy=rsi_buy, rsi_sell=rsi_sell, macd_hist_th=macd_hist_th)

    spark = sparkline_svg(ind['Close'].tail(120))

    options = ["EURUSD=X","GBPUSD=X","USDZAR=X","USDJPY=X","AUDUSD=X","USDCAD=X","AAPL","MSFT","SPY","TSLA"]
    opts_html = "".join([f"<option value='{o}' {'selected' if o==pair else ''}>{o}</option>" for o in options])

    html = f"""
    <html>
    <head>
      <meta name='viewport' content='width=device-width, initial-scale=1' />
      <title>Signal Dashboard — {pair}</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#0a0f1e; color:#e6edf3; margin:0; padding:24px; }}
        .card {{ background:#0f162e; border:1px solid #1f2a44; border-radius:16px; padding:20px; max-width:980px; margin:auto; box-shadow:0 10px 30px rgba(0,0,0,0.2);}}
        h1 {{ margin-top:0; font-size:22px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-top:16px; }}
        .metric {{ background:#0b1226; border:1px solid #1d2844; padding:12px; border-radius:12px; }}
        .badge {{ display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; }}
        .buy {{ background:#153f2a; color:#33d17a; border:1px solid #1f6f47; }}
        .sell {{ background:#4a1b1b; color:#ff6b6b; border:1px solid #7a2e2e; }}
        .hold {{ background:#283143; color:#8fb0ff; border:1px solid #3b4a69; }}
        .spark {{ margin: 12px 0; }}
        .form {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin:10px 0 16px; }}
        label {{ font-size:12px; opacity:.8; }}
        input, select {{ width:100%; padding:8px; border-radius:8px; border:1px solid #24304d; background:#0b1226; color:#e6edf3; }}
        button {{ background:#1b2540; color:#e6edf3; border:1px solid #2d3a5c; padding:10px 14px; border-radius:10px; cursor:pointer; }}
        button:hover {{ background:#232f52; }}
        .reasons li {{ margin-bottom:6px; }}
        .footer {{ opacity:.65; font-size:12px; margin-top:14px; }}
        a, a:visited {{ color:#8fb0ff; }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Signal Dashboard — {pair} <span class="badge {'buy' if sig=='BUY' else 'sell' if sig=='SELL' else 'hold'}">{sig}</span></h1>

        <form class="form" method="get" action="/dashboard">
          <div>
            <label>Symbol</label>
            <select name="pair">{opts_html}</select>
          </div>
          <div>
            <label>Interval</label>
            <select name="interval">
              {''.join([f"<option {'selected' if interval==v else ''} value='{v}'>{v}</option>" for v in ['1h','4h','1d','1wk']])}
            </select>
          </div>
          <div>
            <label>Period</label>
            <select name="period">
              {''.join([f"<option {'selected' if period==v else ''} value='{v}'>{v}</option>" for v in ['30d','60d','90d','1y','2y']])}
            </select>
          </div>
          <div><label>EMA Fast</label><input type="number" name="ema_fast" value="{ema_fast}" min="2" max="200"></div>
          <div><label>EMA Slow</label><input type="number" name="ema_slow" value="{ema_slow}" min="3" max="400"></div>
          <div><label>RSI Length</label><input type="number" name="rsi_len" value="{rsi_len}" min="2" max="200"></div>
          <div><label>RSI Buy&lt;=</label><input type="number" name="rsi_buy" value="{rsi_buy}" min="0" max="100" step="0.5"></div>
          <div><label>RSI Sell&gt;=</label><input type="number" name="rsi_sell" value="{rsi_sell}" min="0" max="100" step="0.5"></div>
          <div><label>MACD Hist |th|</label><input type="number" name="macd_hist_th" value="{macd_hist_th}" min="0" max="5" step="0.01"></div>
          <div style="align-self:end;"><button type="submit">Update</button></div>
          <div style="align-self:end;"><a href="/signal?pair={pair}&interval={interval}&period={period}&ema_fast={ema_fast}&ema_slow={ema_slow}&rsi_len={rsi_len}&rsi_buy={rsi_buy}&rsi_sell={rsi_sell}&macd_fast={macd_fast}&macd_slow={macd_slow}&macd_signal={macd_signal}&macd_hist_th={macd_hist_th}" target="_blank">View JSON ↗</a></div>
        </form>

        <div class="spark">{spark}</div>
        <div class="grid">
          <div class="metric"><strong>Close</strong><div>{latest['Close']:.6f}</div></div>
          <div class="metric"><strong>RSI({rsi_len})</strong><div>{latest[f'rsi_{rsi_len}']:.2f}</div></div>
          <div class="metric"><strong>EMA Fast({ema_fast})</strong><div>{latest['ema_fast']:.6f}</div></div>
          <div class="metric"><strong>EMA Slow({ema_slow})</strong><div>{latest['ema_slow']:.6f}</div></div>
          <div class="metric"><strong>MACD</strong><div>{latest['macd']:.6f}</div></div>
          <div class="metric"><strong>MACD Signal</strong><div>{latest['macd_signal']:.6f}</div></div>
          <div class="metric"><strong>MACD Hist</strong><div>{latest['macd_hist']:.6f}</div></div>
          <div class="metric"><strong>BB Lower</strong><div>{latest['bb_lower']:.6f}</div></div>
          <div class="metric"><strong>BB Mid</strong><div>{latest['bb_mid']:.6f}</div></div>
          <div class="metric"><strong>BB Upper</strong><div>{latest['bb_upper']:.6f}</div></div>
          <div class="metric"><strong>ATR(14)</strong><div>{latest['atr_14']:.6f}</div></div>
        </div>
        <h3>Reasons</h3>
        <ul class="reasons">
          {''.join(f'<li>{r}</li>' for r in reasons)}
        </ul>
        <div class="footer">
          Tip: Change symbol to <code>GBPUSD=X</code> or <code>USDZAR=X</code>, switch interval to <code>1d</code>, and adjust RSI thresholds.
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

@app.post("/webhook", response_class=HTMLResponse)
async def webhook(request: Request):
    """
    Forwards posted JSON to WEBHOOK_URL if set; otherwise echoes it back.
    """
    data = await request.json()
    url = os.getenv("WEBHOOK_URL", "").strip()
    status = "skipped (WEBHOOK_URL not set)"
    if url:
        try:
            requests.post(url, json=data, timeout=5)
            status = "forwarded"
        except Exception as e:
            status = f"error forwarding: {e}"
    return {"status": status, "received": data}
