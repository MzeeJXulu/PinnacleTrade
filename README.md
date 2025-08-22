# Pandas-TA Signal Service (Render-ready)

Minimal FastAPI service that computes a simple trading signal using `pandas-ta`.
Built to avoid compiling `pandas` on Render by pinning versions with Linux wheels.

## Endpoints
- `GET /` health check
- `GET /signal?ticker=SPY&interval=1d&period=60d` Compute RSI/EMA-based signal

## Local dev
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Render
- Use the included `render.yaml`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Runtime: Python 3.11.9 (via `runtime.txt` or Render env var)
