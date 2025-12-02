# Backend - Market Regime Explorer API

FastAPI backend for market regime detection and next-day SPY direction prediction.

## Local Development

### Setup

```bash
# From repo root
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Run

```bash
# From repo root
uvicorn backend.api:app --reload

# Or from inside backend/ directory
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Environment Variables

- `OPENAI_API_KEY` (optional, required for `/llm/explain` endpoint)

Create a `.env` file in the `backend/` directory or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Railway Deployment

### Configuration

- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Environment Variables

Set in Railway dashboard:

- `OPENAI_API_KEY` (required for `/llm/explain` endpoint)

### Notes

- Railway will automatically set `$PORT` environment variable
- The backend will download data from Yahoo Finance on first startup if `data/stocks_raw.csv` doesn't exist
- Daily refresh is scheduled for 22:30 UTC via APScheduler

## API Endpoints

- `GET /` - Health check
- `GET /summary` - Model metrics, cluster stats, latest predictions
- `GET /timeseries` - Time series data for test period
- `GET /pca_scatter` - PCA scatter plot data
- `POST /refresh` - Manually trigger pipeline refresh
- `POST /llm/explain` - LLM-powered explanation (requires OPENAI_API_KEY)

