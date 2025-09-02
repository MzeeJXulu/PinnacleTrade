# app/main.py
from __future__ import annotations

import csv
import io
import os
import re
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Your indicator logic lives here:
# - compute_indicators(df, ema_fast, ema_slow, rsi_len, macd_fast, macd_slow, macd_signal) -> pd.DataFrame
# - decide_signal(latest_row, rsi_len, rsi_buy, rsi_sell, macd_hist_th) -> (signal_str, reasons_list)
from app.analysis_engine import compute_indicators, decide_signal

app = FastAPI(title="PinnacleTrade — Signal Service", version="3.3.0")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _user_agent_headers() -> Dict[str, str]:
    # Some endpoints return blank unless a browser-like UA is present
    return {"User-Agent": "Mozilla/5.0 (compatible; PinnacleTrade/1.0; +https://pinnacletrade.onrender.com)"}


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly Open, High, Low, Close, Volume (float)."""
    if df is None or df.empty:
        return pd.DataFrame()
    # handle both lowercase & standard columns
    cols = {c.lower(): c for c in df.columns}
    want = ["open", "high", "low", "close"]
    if all(w in cols for w in want):
        out = df.rename(
            columns={
                cols["open"]: "Open",
                cols["high"]: "High",
                cols["low"]: "Low",
                cols["close"]: "Close",
            }
        ).copy()
        out["Volume"] = df[cols["volume"]] if "volume" in cols else 0.0
        return out[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    # otherwise assume already standard
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


# ---------------------------------------------------------------------
# Provider: Yahoo Finance
# ---------------------------------------------------------------------
def _yahoo_fetch(pair: str, interval: str, period: str, debug: dict) -> pd.DataFrame:
    """
    Try a few combos, because Yahoo can be temperamental on server hosts.
    """
    combos = [
        (interval, period),
        ("1d", "1y"),
        ("1d", "2y"),
        ("1wk", "5y"),
    ]
    for iv, pr in combos:
        try:
            # Note: threads=False for stability on Render
            df = yf.download(pair, interval=iv, period=pr, progress=False, threads=False)
            debug.setdefault("yahoo_attempts", []).append(
                {"interval": iv, "period": pr, "rows": 0 if df is None else int(df.shape[0])}
            )
            if df is not None and not df.empty:
                df.attrs["used_interval"] = iv
                df.attrs["used_period"] = pr
                df.attrs["used_provider"] = "yahoo"
                return _standardize_ohlcv(df)
        except Exception as e:
            debug.setdefault("yahoo_errors", []).append({"interval": iv, "period": pr, "error": str(e)})
    return pd.DataFrame()


# ---------------------------------------------------------------------
# Provider: Alpha Vantage (optional, needs API key)
# ---------------------------------------------------------------------
def _parse_fx_pair_for_av(pair: str) -> tuple[str, str]:
    p = pair.upper().strip()
    if "=X" in p and len(p) >= 8:
        core = p.replace("=X", "")
        if len(core) >= 6:
            return core[:3], core[3:6]
    if "/" in p:
        a, b = p.split("/", 1)
        if len(a) == 3 and len(b) == 3:
            return a, b
    if re.fullmatch(r"[A-Z]{6}", p):
        return p[:3], p[3:]
    raise ValueError(f"Cannot parse FX pair for Alpha Vantage: {pair}")


def _alpha_vantage_fetch(pair: str, interval: str, period: str, api_key: str, debug: dict) -> pd.DataFrame:
    if not api_key:
        debug["alpha_vantage"] = "skipped (no API key)"
        return pd.DataFrame()

    try:
        frm, to = _parse_fx_pair_for_av(pair)
    except Exception as e:
        debug["alpha_vantage_parse"] = str(e)
        return pd.DataFrame()

    def _av_intraday(mins: str, outputsize: str = "compact") -> pd.DataFrame:
        url = "https://www.alphavantage.co/query"
        params = dict(
            function="FX_INTRADAY",
            from_symbol=frm,
            to_symbol=to,
            interval=f"{mins}min",
            outputsize=outputsize,
            apikey=api_key,
        )
        r = requests.get(url, params=params, timeout=30, headers=_user_agent_headers())
        j = r.json()
        key = f"Time Series FX ({mins}min)"
        if key not in j:
            debug.setdefault("alpha_vantage_resp", []).append(
                {"endpoint": "intraday", "interval": mins, "note": list(j.keys())[:5]}
            )
            return pd.DataFrame()
        rec = []
        for ts, row in j[key].items():
            rec.append(
                {
                    "timestamp": pd.to_datetime(ts),
                    "Open": float(row["1. open"]),
                    "High": float(row["2. high"]),
                    "Low": float(row["3. low"]),
                    "Close": float(row["4. close"]),
                    "Volume": 0.0,
                }
            )
        df = pd.DataFrame(rec).sort_values("timestamp").set_index("timestamp")
        df.attrs["used_provider"] = "alpha_vantage"
        return df

    def _av_daily(outputsize: str = "compact") -> pd.DataFrame:
        url = "https://www.alphavantage.co/query"
        params = dict(function="FX_DAILY", from_symbol=frm, to_symbol=to, outputsize=outputsize, apikey=api_key)
        r = requests.get(url, params=params, timeout=30, headers=_user_agent_headers())
        j = r.json()
        key = "Time Series FX (Daily)"
        if key not in j:
            debug.setdefault("alpha_vantage_resp", []).append({"endpoint": "daily", "note": list(j.keys())[:5]})
            return pd.DataFrame()
        rec = []
        for ts, row in j[key].items():
            rec.append(
                {
                    "timestamp": pd.to_datetime(ts),
                    "Open": float(row["1. open"]),
                    "High": float(row["2. high"]),
                    "Low": float(row["3. low"]),
                    "Close": float(row["4. close"]),
                    "Volume": 0.0,
                }
            )
        df = pd.DataFrame(rec).sort_values("timestamp").set_index("timestamp")
        df.attrs["used_provider"] = "alpha_vantage"
        return df

    long = period in {"1y", "2y", "5y", "max", "60d", "90d"}
    outputsize = "full" if long else "compact"

    try:
        if interval == "1h":
            return _av_intraday("60", outputsize)
        if interval == "4h":
            base = _av_intraday("60", outputsize)
            if base.empty:
                return base
            out = (
                base.resample("4H")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )
            out.attrs["used_provider"] = "alpha_vantage"
            out.attrs["used_interval"] = "4h"
            return out
        if interval == "1d":
            return _av_daily(outputsize)
        if interval == "1wk":
            base = _av_daily(outputsize)
            if base.empty:
                return base
            out = (
                base.resample("W")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )
            out.attrs["used_provider"] = "alpha_vantage"
            out.attrs["used_interval"] = "1wk"
            return out
        # default: daily
        return _av_daily(outputsize)
    except Exception as e:
        debug["alpha_vantage_error"] = str(e)
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Provider: Stooq (free CSV; daily FX)
# ---------------------------------------------------------------------
def _stooq_symbol(pair: str) -> Optional[str]:
    """
    Map common FX tickers to Stooq symbol (lowercase, no '=X').
    Examples:
      EURUSD=X -> eurusd
      USDZAR=X -> usdzar
      GBPUSD=X -> gbpusd
    """
    p = pair.upper().strip()
    if p.endswith("=X") and len(p) >= 8:
        core = p.replace("=X", "")
        if re.fullmatch(r"[A-Z]{6}", core):
            return core.lower()
    if re.fullmatch(r"[A-Z]{6}", p):
        return p.lower()
    if "/" in p and len(p.replace("/", "")) == 6:
        return p.replace("/", "").lower()
    return None


def _stooq_fetch(pair: str, interval: str, period: str, debug: dict) -> pd.DataFrame:
    """
    Stooq delivers **daily** FX CSV. Accept either:
      - Date,Open,High,Low,Close
      - Date,Open,High,Low,Close,Volume
    """
    sym = _stooq_symbol(pair)
    if not sym:
        debug["stooq"] = "skipped (non-FX or unmapped symbol)"
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=30, headers=_user_agent_headers())
        if r.status_code != 200 or not r.text:
            debug["stooq_status"] = f"HTTP {r.status_code}, empty body"
            return pd.DataFrame()

        reader = csv.reader(io.StringIO(r.text))
        rows = list(reader)
        if not rows:
            debug["stooq_status"] = "no rows"
            return pd.DataFrame()

        header = [h.strip() for h in rows[0]]
        data_rows = rows[1:]

        header_lc = [h.lower() for h in header]
        expected = ["date", "open", "high", "low", "close"]
        if not all(col in header_lc for col in expected):
            debug["stooq_status"] = f"unexpected header: {header[:6]}"
            return pd.DataFrame()

        idx_date = header_lc.index("date")
        idx_open = header_lc.index("open")
        idx_high = header_lc.index("high")
        idx_low = header_lc.index("low")
        idx_close = header_lc.index("close")
        idx_vol = header_lc.index("volume") if "volume" in header_lc else None

        rec = []
        for row in data_rows:
            if not row or len(row) < 5:
                continue
            try:
                rec.append(
                    {
                        "timestamp": pd.to_datetime(row[idx_date]),
                        "Open": float(row[idx_open]),
                        "High": float(row[idx_high]),
                        "Low": float(row[idx_low]),
                        "Close": float(row[idx_close]),
                        "Volume": float(row[idx_vol]) if idx_vol is not None and row[idx_vol] not in (None, "") else 0.0,
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rec).sort_values("timestamp").set_index("timestamp")
        debug["stooq_rows"] = int(df.shape[0])

        # Stooq is daily — allow simple weekly resample on request
        if interval == "1wk" and not df.empty:
            df = (
                df.resample("W")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )
            df.attrs["used_interval"] = "1wk"
        else:
            df.attrs["used_interval"] = "1d"

        df.attrs["used_period"] = "5y"
        df.attrs["used_provider"] = "stooq"
        return df

    except Exception as e:
        debug["stooq_error"] = str(e)
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------
def fetch_market_data(pair: str, interval: str, period: str, debug: Optional[dict] = None) -> pd.DataFrame:
    if debug is None:
        debug = {}
    provider_order = []
    provider = os.getenv("DATA_PROVIDER", "yahoo").strip().lower()
    key = os.getenv("ALPHA_VANTAGE_KEY", "").strip()

    # If user forces alpha_vantage, try that first, then yahoo, then stooq
    if provider == "alpha_vantage":
        provider_order.append("alpha_vantage")
        df = _alpha_vantage_fetch(pair, interval, period, key, debug)
        if not df.empty:
            return df

        provider_order.append("yahoo")
        df = _yahoo_fetch(pair, interval, period, debug)
        if not df.empty:
            return df

        provider_order.append("stooq")
        df = _stooq_fetch(pair, interval, period, debug)
        if not df.empty:
            return df
    else:
        # Default: yahoo → stooq → (alpha if key present)
        provider_order.append("yahoo")
        df = _yahoo_fetch(pair, interval, period, debug)
        if not df.empty:
            return df

        provider_order.append("stooq")
        df = _stooq_fetch(pair, interval, period, debug)
        if not df.empty:
            return df

        if key:
            provider_order.append("alpha_vantage")
            df = _alpha_vantage_fetch(pair, interval, period, key, debug)
            if not df.empty:
                return df

    debug["provider_order"] = provider_order
    return pd.DataFrame()


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
def sparkline_svg(series, width=720, height=160, pad=12) -> str:
    vals = [float(x) for x in series if pd.notna(x)]
    if not vals or len(vals) < 2:
        return f"<svg width='{width}' height='{height}'></svg>"
    vmin, vmax = min(vals), max(vals)
    vrange = (vmax - vmin) or 1.0
    step = (width - 2 * pad) / (len(vals) - 1)
    pts = []
    for i, v in enumerate(vals):
        x = pad + i * step
        y = pad + (height - 2 * pad) * (1 - (v - vmin) / vrange)
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


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/debug")
def debug_endpoint(
    pair: str = Query("EURUSD=X"),
    interval: str = Query("1h"),
    period: str = Query("60d"),
):
    dbg = {}
    df = fetch_market_data(pair, interval, period, debug=dbg)
    info = {
        "env_DATA_PROVIDER": os.getenv("DATA_PROVIDER", "yahoo"),
        "used_provider": getattr(df, "used_provider", None),
        "used_interval": getattr(df, "used_interval", None),
        "used_period": getattr(df, "used_period", None),
        "rows": None if df is None else int(df.shape[0]),
        "head": None if df is None or df.empty else df.head(3).reset_index().to_dict(orient="records"),
        "tail": None if df is None or df.empty else df.tail(3).reset_index().to_dict(orient="records"),
        "trace": dbg,
    }
    return JSONResponse(info)


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
    dbg = {}
    df: pd.DataFrame = fetch_market_data(pair, interval, period, debug=dbg)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data returned (exhausted). trace={dbg}")

    ind = compute_indicators(
        df,
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
        latest, rsi_len=rsi_len, rsi_buy=rsi_buy, rsi_sell=rsi_sell, macd_hist_th=macd_hist_th
    )

    used_provider = getattr(df, "used_provider", os.getenv("DATA_PROVIDER", "yahoo"))
    used_interval = getattr(df, "used_interval", interval)
    used_period = getattr(df, "used_period", period)

    payload = {
        "pair": pair,
        "interval": used_interval,
        "period": used_period,
        "provider": used_provider,
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
        "trace": dbg,
    }

    # Optional: forward to your webhook (set WEBHOOK_URL in Render)
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
    dbg = {}
    df: pd.DataFrame = fetch_market_data(pair, interval, period, debug=dbg)
    if df is None or df.empty:
        return HTMLResponse(f"<h2>No data returned (exhausted)</h2><pre>{dbg}</pre>", status_code=404)

    ind = compute_indicators(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_len=rsi_len,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    ).dropna()
    if ind.empty:
        return HTMLResponse("<h2>Not enough data after indicators</h2>", status_code=422)

    latest = ind.iloc[-1]
    sig, reasons = decide_signal(
        latest, rsi_len=rsi_len, rsi_buy=rsi_buy, rsi_sell=rsi_sell, macd_hist_th=macd_hist_th
    )

    spark = sparkline_svg(ind["Close"].tail(120))
    options = ["EURUSD=X", "GBPUSD=X", "USDZAR=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "AAPL", "MSFT", "SPY", "TSLA"]
    opts_html = "".join([f"<option value='{o}' {'selected' if o == pair else ''}>{o}</option>" for o in options])
    used_interval = getattr(df, "used_interval", interval)
    used_period = getattr(df, "used_period", period)
    provider = getattr(df, "used_provider", os.getenv("DATA_PROVIDER", "yahoo"))

    html = f"""
    <html><head>
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
        pre {{ white-space: pre-wrap; font-size:12px; opacity:.8; }}
      </style>
    </head><body>
      <div class="card">
        <h1>
          Signal Dashboard — {pair}
          <small style="opacity:.6;">[{provider} • {used_interval}, {used_period}]</small>
          <span class="badge {'buy' if sig=='BUY' else 'sell' if sig=='SELL' else 'hold'}">{sig}</span>
        </h1>

        <form class="form" method="get" action="/dashboard">
          <div><label>Symbol</label><select name="pair">{opts_html}</select></div>
          <div><label>Interval</label><select name="interval">
            {''.join([f"<option {'selected' if interval==v else ''} value='{v}'>{v}</option>" for v in ['1h','4h','1d','1wk']])}
          </select></div>
          <div><label>Period</label><select name="period">
            {''.join([f"<option {'selected' if period==v else ''} value='{v}'>{v}</option>" for v in ['30d','60d','90d','1y','2y']])}
          </select></div>
          <div><label>EMA Fast</label><input type="number" name="ema_fast" value="{ema_fast}" min="2" max="200"></div>
          <div><label>EMA Slow</label><input type="number" name="ema_slow" value="{ema_slow}" min="3" max="400"></div>
          <div><label>RSI Length</label><input type="number" name="rsi_len" value="{rsi_len}" min="2" max="200"></div>
          <div><label>RSI Buy&lt;=</label><input type="number" name="rsi_buy" value="{rsi_buy}" min="0" max="100" step="0.5"></div>
          <div><label>RSI Sell&gt;=</label><input type="number" name="rsi_sell" value="{rsi_sell}" min="0" max="100" step="0.5"></div>
          <div><label>MACD Hist |th|</label><input type="number" name="macd_hist_th" value="{macd_hist_th}" min="0" max="5" step="0.01"></div>
          <div style="align-self:end;"><button type="submit">Update</button></div>
          <div style="align-self:end;"><a href="/signal?pair={pair}&interval={interval}&period={period}&ema_fast={ema_fast}&ema_slow={ema_slow}&rsi_len={rsi_len}&rsi_buy={rsi_buy}&rsi_sell={rsi_sell}&macd_fast={macd_fast}&macd_slow={macd_slow}&macd_signal={macd_signal}&macd_hist_th={macd_hist_th}" target="_blank">View JSON ↗</a></div>
          <div style="align-self:end;"><a href="/debug?pair={pair}&interval={interval}&period={period}" target="_blank">Debug fetch ↗</a></div>
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
        <ul class="reasons">{''.join(f'<li>{r}</li>' for r in reasons)}</ul>
        <div class="footer">Tip: for reliable intraday (1h/4h), set DATA_PROVIDER=alpha_vantage and ALPHA_VANTAGE_KEY in Render.</div>
        <details><summary>Fetch trace</summary><pre>{dbg}</pre></details>
      </div>
    </body></html>
    """
    return HTMLResponse(content=html, status_code=200)


@app.post("/webhook", response_class=HTMLResponse)
async def webhook(request: Request):
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
