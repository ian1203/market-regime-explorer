# Market Regime Explorer

End-to-end ML pipeline and interactive dashboard for stock market regime detection and next-day SPY direction prediction.

## Overview

Market Regime Explorer is a comprehensive machine learning project that analyzes daily price data from multiple ETFs (SPY, QQQ, DIA, IWM, XLK, XLF, XLV) to identify distinct market regimes and predict whether the SPY ETF will close up or down on the next trading day.

The system:
- Downloads historical data from Yahoo Finance for multiple ETFs
- Engineers 100+ features including lagged returns, rolling statistics, and volume-based indicators
- Applies dimensionality reduction with PCA (20 components) and unsupervised clustering with K-Means (K=2) to discover market regimes
- Trains multiple classification models (Logistic Regression and Random Forest) on both full features and PCA-reduced features
- Provides a FastAPI backend with RESTful endpoints and a Next.js dashboard with interactive visualizations
- Includes an LLM-powered AI assistant that explains market regimes and model predictions in plain language

## Project Structure

```
market-regime-explorer/
├── backend/          # FastAPI API + ML pipeline
│   ├── api.py        # FastAPI application
│   ├── models.py     # ML pipeline functions
│   └── requirements.txt
├── frontend/         # Next.js dashboard UI
│   ├── components/   # React components
│   ├── pages/        # Next.js pages
│   ├── lib/          # API client and utilities
│   └── styles/       # CSS and Tailwind config
├── data/             # CSV with historical prices
├── notebooks/        # Exploration notebooks
└── README.md
```

## Features

- **Time-based Data Splitting**: Proper train/validation/test split to avoid data leakage
- **Dimensionality Reduction**: PCA with 20 components to capture most variance while reducing noise
- **Market Regime Detection**: K-Means clustering (K=2) on PCA space to identify distinct market regimes
- **Model Comparison**: Multiple models evaluated with comprehensive metrics (Accuracy, F1-score, ROC-AUC)
- **Interactive Dashboard**: SPY price vs predicted probability, PCA scatter plots, regime statistics, and LLM-powered AI insights
- **Self-Updating Backend**: Automatic daily refresh at 22:30 UTC via APScheduler, plus manual refresh endpoint
- **Production-Ready API**: FastAPI with CORS, error handling, and structured responses

## Tech Stack

### Backend
- Python 3.9+, FastAPI, pandas, numpy, scikit-learn, yfinance, OpenAI API, APScheduler

### Frontend
- Next.js 14, React 18, TypeScript, TailwindCSS, Recharts

## Getting Started

### Backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the backend
uvicorn backend.api:app --reload
```

The backend will be available at `http://localhost:8000`

**Environment Variables** (optional, for LLM endpoint):
- `OPENAI_API_KEY` - Required for `/llm/explain` endpoint

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

**Environment Variables** (optional):
- `NEXT_PUBLIC_API_BASE_URL` - Backend API URL (defaults to `http://localhost:8000`)

## Deployment

### Backend → Railway

1. Connect your GitHub repository to Railway
2. Create a new service and point it to the `backend/` directory
3. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Set environment variables:
   - `OPENAI_API_KEY` (required for `/llm/explain`)

### Frontend → Vercel

1. Connect your GitHub repository to Vercel
2. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Next.js
3. Set environment variables:
   - `NEXT_PUBLIC_API_BASE_URL` - Your Railway backend URL (e.g., `https://your-app.railway.app`)

## API Endpoints

- `GET /` - Health check
- `GET /summary` - Model metrics, cluster stats, latest predictions
- `GET /timeseries` - Time series data for test period
- `GET /pca_scatter` - PCA scatter plot data
- `POST /refresh` - Manually trigger pipeline refresh
- `POST /llm/explain` - LLM-powered explanation (requires OPENAI_API_KEY)

See `backend/README_BACKEND.md` for detailed backend documentation.

## Notes for Recruiters / Professors

This project demonstrates a complete machine learning lifecycle from data ingestion to production deployment:

- **Data Engineering**: Automated data download, feature engineering (100+ features), and proper time-series splitting
- **Dimensionality Reduction**: PCA for noise reduction and visualization
- **Unsupervised Learning**: K-Means clustering to discover latent market regimes
- **Supervised Learning**: Multiple model comparison with proper evaluation metrics
- **Backend Development**: RESTful API with FastAPI, scheduled tasks, and error handling
- **Frontend Development**: Modern React/Next.js dashboard with TypeScript and responsive design
- **LLM Integration**: OpenAI API integration for model interpretability and user-friendly explanations
- **Production Practices**: Environment variable management, CORS configuration, logging, and automated data refresh

The project showcases skills in:
- Python data science stack (pandas, scikit-learn, numpy)
- Modern web development (React, TypeScript, Next.js)
- API design and development (FastAPI, REST)
- Machine learning best practices (train/val/test splits, feature engineering, model evaluation)
- Software engineering (modular code, type hints, error handling)

## License

This project is for educational and portfolio purposes.
