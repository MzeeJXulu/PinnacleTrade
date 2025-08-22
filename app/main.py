from fastapi import FastAPI
from app.analysis_engine import generate_signal

app = FastAPI()

@app.get("/signal")
def signal(pair: str = "EURUSD"):
    return generate_signal(pair)
