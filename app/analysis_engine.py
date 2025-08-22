import yfinance as yf
import pandas as pd
import pandas_ta as ta

def fetch_data(pair="EURUSD=X", period="1mo", interval="1h"):
    df = yf.download(pair, period=period, interval=interval)
    df = df.rename(columns=str.lower)
    return df

def add_technical_indicators(df):
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["sma"] = ta.sma(df["close"], length=20)
    df["ema"] = ta.ema(df["close"], length=20)
    return df

def generate_signal(pair="EURUSD=X"):
    df = fetch_data(pair)
    df = add_technical_indicators(df)
    latest = df.iloc[-1]

    if latest["rsi"] < 30 and latest["close"] > latest["ema"]:
        return {"pair": pair, "signal": "BUY"}
    elif latest["rsi"] > 70 and latest["close"] < latest["ema"]:
        return {"pair": pair, "signal": "SELL"}
    else:
        return {"pair": pair, "signal": "HOLD"}
