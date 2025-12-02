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

## Features

- **Time-based Data Splitting**: Proper train/validation/test split to avoid data leakage
  - Training: ≤ 2018-12-31
  - Validation: 2019-01-02 to 2021-12-31
  - Test: 2022-01-03 onward

- **Dimensionality Reduction**: PCA with 20 components to capture most variance while reducing noise

- **Market Regime Detection**: K-Means clustering (K=2) on PCA space to identify distinct market regimes

- **Model Comparison**: Multiple models evaluated with comprehensive metrics
  - Logistic Regression (full features vs PCA-20)
  - Random Forest (full features vs PCA-20)
  - Metrics: Accuracy, F1-score, ROC-AUC

- **Interactive Dashboard**:
  - SPY price vs predicted "up" probability over the test period
  - PCA scatter plots colored by regime and true label
  - Regime statistics table (mean returns, probability of up days)
  - Model performance metrics table
  - LLM-powered "AI Insight" panel that explains the current regime and predictions

- **Self-Updating Backend**: Automatic daily refresh at 22:30 UTC via APScheduler, plus manual refresh endpoint

- **Production-Ready API**: FastAPI with CORS, error handling, and structured responses

## Tech Stack

### Backend
- **Python 3.9+**
- **FastAPI**: RESTful API framework
- **pandas & numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning (PCA, K-Means, Logistic Regression, Random Forest)
- **yfinance**: Yahoo Finance data download
- **OpenAI API**: LLM-powered explanations (gpt-4o-mini)
- **APScheduler**: Scheduled daily data refresh
- **uvicorn**: ASGI server

### Frontend
- **Next.js 14** (Pages Router)
- **React 18** with TypeScript
- **TailwindCSS**: Utility-first CSS framework (dark theme)
- **Recharts**: Data visualization library

### Deployment
- Backend: Deployable on Render, Railway, or similar (requires `OPENAI_API_KEY` environment variable)
- Frontend: Deployable on Vercel, Netlify, or similar

## Project Structure

```
market-regime-explorer/
├── backend/
│   ├── api.py          # FastAPI application with endpoints
│   ├── models.py       # ML pipeline functions
│   └── __init__.py
├── data/
│   └── stocks_raw.csv  # Historical data snapshot
├── notebooks/
│   ├── pipeline.ipynb # Jupyter notebook (original development)
│   └── pipeline.py     # Exported notebook code
├── pages/
│   └── index.tsx      # Main dashboard page
├── components/
│   ├── Layout.tsx
│   ├── SummaryCards.tsx
│   ├── TimeSeriesChart.tsx
│   ├── PCAScatter.tsx
│   ├── MetricsTable.tsx
│   └── ClusterStatsTable.tsx
├── lib/
│   └── api/
│       └── client.ts   # API client with TypeScript types
├── styles/
│   └── globals.css    # Global styles and Tailwind setup
├── requirements.txt   # Python dependencies
├── package.json       # Node.js dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Node.js 18+ and npm/yarn
- OpenAI API key (optional, required for `/llm/explain` endpoint)

### Backend Setup

1. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variable** (optional, for LLM endpoint):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file (not tracked in git).

4. **Run the backend**:
   ```bash
   uvicorn backend.api:app --reload --port 8000
   ```

   The backend will:
   - Load existing data from `data/stocks_raw.csv`
   - Run the full ML pipeline on startup
   - Schedule daily refresh at 22:30 UTC
   - Expose API endpoints at `http://localhost:8000`

### Frontend Setup

1. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

2. **Set environment variable** (optional, if backend is not on localhost:8000):
   ```bash
   # Create .env.local
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open in browser**:
   Navigate to `http://localhost:3000` (or the port shown in terminal)

## API Endpoints

### `GET /`
Health check endpoint.

**Response**:
```json
{
  "message": "Market Regime Explorer API",
  "status": "running"
}
```

### `GET /summary`
Returns summary information including best model, metrics, cluster statistics, and latest predictions.

**Response**:
```json
{
  "best_model_name": "rf_pca",
  "metrics": [
    {
      "model": "logreg_full",
      "accuracy": 0.49,
      "f1": 0.28,
      "roc_auc": 0.48
    },
    ...
  ],
  "cluster_stats": [
    {
      "cluster_id": 0,
      "n_days": 1200,
      "mean_daily_return": 0.0005,
      "prob_up": 0.58
    },
    ...
  ],
  "latest_date": "2024-12-31",
  "latest_regime": 1,
  "latest_pred_prob": 0.537
}
```

### `GET /timeseries`
Returns time series data for the test period.

**Response**:
```json
{
  "data": [
    {
      "date": "2022-01-03",
      "spy_price": 470.3,
      "true_up": 1,
      "pred_up_prob": 0.52,
      "cluster": 0
    },
    ...
  ]
}
```

### `GET /pca_scatter`
Returns sampled PCA data for visualization (up to 2000 points).

**Response**:
```json
{
  "data": [
    {
      "pc1": -2.3,
      "pc2": 1.4,
      "cluster_id": 0,
      "target_up": 1
    },
    ...
  ]
}
```

### `POST /refresh`
Manually trigger a full pipeline refresh (downloads fresh data and re-runs pipeline).

**Response**:
```json
{
  "status": "ok"
}
```

### `POST /llm/explain`
Get LLM-powered explanation of current market regime and predictions.

**Request**:
```json
{
  "question": "What does the current regime mean?"  // Optional
}
```

**Response**:
```json
{
  "answer": "The current market regime (Cluster 1) is characterized by..."
}
```

**Note**: Requires `OPENAI_API_KEY` environment variable to be set on the backend.

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

## Contributing

This is a personal portfolio project. Feel free to fork and adapt for your own learning!

