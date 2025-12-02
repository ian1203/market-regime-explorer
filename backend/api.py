"""
FastAPI application for market regime detection and prediction API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from openai import OpenAI

from backend.models import run_full_pipeline, download_full_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress harmless joblib resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Initialize FastAPI app
app = FastAPI(title="Market Regime Explorer API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level variables to store pipeline results
PIPELINE_RESULT: Optional[Dict] = None
BEST_MODEL = None
SCALER = None
PCA = None
KMEANS = None
CLOSE_DF = None
FEATURES_DF = None
SPLITS = None

# Initialize scheduler (will be started in on_startup)
scheduler = AsyncIOScheduler()

# Initialize OpenAI client (reads from environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _reload_pipeline(reload_data: bool = False) -> None:
    """
    Internal helper: optionally re-download data, then re-run the full pipeline
    and update module-level globals.
    
    Args:
        reload_data: If True, download fresh data from Yahoo Finance before running pipeline.
    """
    global PIPELINE_RESULT, BEST_MODEL, SCALER, PCA, KMEANS, CLOSE_DF, FEATURES_DF, SPLITS
    
    try:
        project_root = Path(__file__).parent.parent
        csv_path = project_root / "data" / "stocks_raw.csv"
        
        if reload_data:
            logger.info("Downloading fresh data from Yahoo Finance...")
            download_full_history(str(csv_path))
            logger.info("Data download complete. Running pipeline...")
        else:
            logger.info(f"Loading pipeline from existing CSV: {csv_path}")
        
        # Run full pipeline
        result = run_full_pipeline(str(csv_path))
        
        # Atomically update all module-level variables
        PIPELINE_RESULT = result
        BEST_MODEL = result["models_info"]["best_model"]
        SCALER = result["splits"]["scaler"]
        PCA = result["splits"]["pca"]
        KMEANS = result["kmeans"]
        CLOSE_DF = result["close"]
        FEATURES_DF = result["features"]
        SPLITS = result["splits"]
        
        logger.info("Pipeline reload complete successfully.")
        
    except Exception as e:
        logger.error(f"Error reloading pipeline: {e}", exc_info=True)
        raise


def scheduled_refresh_job():
    """
    Job run by APScheduler to refresh data and pipeline daily.
    Safe to call the same helper used by /refresh.
    """
    logger.info("Scheduled daily refresh job triggered.")
    # We use reload_data=True so it re-downloads from Yahoo Finance.
    _reload_pipeline(reload_data=True)


# Pydantic models for responses
class ModelMetric(BaseModel):
    model: str
    accuracy: float
    f1: float
    roc_auc: float


class ClusterStat(BaseModel):
    cluster_id: int
    n_days: int
    mean_daily_return: float
    prob_up: float


class SummaryResponse(BaseModel):
    best_model_name: str
    metrics: List[ModelMetric]
    cluster_stats: List[ClusterStat]
    latest_date: str
    latest_regime: int
    latest_pred_prob: float


class TimeSeriesPoint(BaseModel):
    date: str
    spy_price: float
    true_up: int
    pred_up_prob: float
    cluster: int


class TimeSeriesResponse(BaseModel):
    data: List[TimeSeriesPoint]


class PCAScatterPoint(BaseModel):
    pc1: float
    pc2: float
    cluster_id: int
    target_up: int


class PCAScatterResponse(BaseModel):
    data: List[PCAScatterPoint]


class LLMExplainRequest(BaseModel):
    question: Optional[str] = None


class LLMExplainResponse(BaseModel):
    answer: str


class RefreshResponse(BaseModel):
    status: str


@app.on_event("startup")
def on_startup():
    """Initialize pipeline and start scheduler on application startup."""
    # 1. Load pipeline initially (without hitting Yahoo if CSV is already there)
    _reload_pipeline(reload_data=False)
    
    # 2. Configure daily job – e.g. at 22:30 UTC
    trigger = CronTrigger(hour=22, minute=30, timezone="UTC")
    scheduler.add_job(
        scheduled_refresh_job,
        trigger=trigger,
        id="daily_refresh",
        replace_existing=True
    )
    
    # 3. Start scheduler
    scheduler.start()
    logger.info("Scheduler started. Daily refresh scheduled for 22:30 UTC.")


@app.on_event("shutdown")
def on_shutdown():
    """Shutdown scheduler on application shutdown."""
    scheduler.shutdown(wait=False)
    logger.info("Scheduler shut down.")


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Market Regime Explorer API", "status": "running"}


@app.get("/summary", response_model=SummaryResponse)
def get_summary():
    """
    Returns summary information including:
    - Best model name and metrics for all models
    - Cluster statistics
    - Latest available date and its regime/prediction
    """
    if PIPELINE_RESULT is None:
        raise ValueError("Pipeline not loaded. Please restart the application.")
    
    models_info = PIPELINE_RESULT["models_info"]
    cluster_stats = PIPELINE_RESULT["cluster_stats"]
    
    # Get metrics
    metrics_df = models_info["metrics"]
    metrics = [
        ModelMetric(
            model=row["model"],
            accuracy=row["accuracy"],
            f1=row["f1"],
            roc_auc=row["roc_auc"],
        )
        for _, row in metrics_df.iterrows()
    ]
    
    # Get cluster stats
    cluster_stats_list = [
        ClusterStat(
            cluster_id=row["cluster_id"],
            n_days=row["n_days"],
            mean_daily_return=row["mean_daily_return"],
            prob_up=row["prob_up"],
        )
        for _, row in cluster_stats.iterrows()
    ]
    
    # Get latest date from test set
    dates_test = SPLITS["dates_test"]
    latest_date = dates_test[-1]
    latest_date_str = latest_date.strftime("%Y-%m-%d")
    
    # Get latest test index
    latest_idx = len(dates_test) - 1
    
    # Get latest regime (cluster)
    latest_regime = int(PIPELINE_RESULT["cluster_test"][latest_idx])
    
    # Get latest prediction probability using preprocessed features
    X_test_scaled = SPLITS["X_test_scaled"]
    X_test_pca = SPLITS["X_test_pca"]
    
    # Use best model to predict
    if models_info["best_model_name"] in ["logreg_pca", "rf_pca"]:
        pred_prob = float(BEST_MODEL.predict_proba(X_test_pca[latest_idx:latest_idx+1])[0, 1])
    else:
        pred_prob = float(BEST_MODEL.predict_proba(X_test_scaled[latest_idx:latest_idx+1])[0, 1])
    
    return SummaryResponse(
        best_model_name=models_info["best_model_name"],
        metrics=metrics,
        cluster_stats=cluster_stats_list,
        latest_date=latest_date_str,
        latest_regime=latest_regime,
        latest_pred_prob=pred_prob,
    )


@app.get("/timeseries", response_model=TimeSeriesResponse)
def get_timeseries():
    """
    Returns time series over the test period with:
    - date
    - spy_price (SPY close)
    - true_up (0/1)
    - pred_up_prob (probability from best model)
    - cluster (regime id)
    """
    if PIPELINE_RESULT is None:
        raise ValueError("Pipeline not loaded. Please restart the application.")
    
    dates_test = SPLITS["dates_test"]
    y_test = SPLITS["y_test"]
    X_test_scaled = SPLITS["X_test_scaled"]
    X_test_pca = SPLITS["X_test_pca"]
    cluster_test = PIPELINE_RESULT["cluster_test"]
    models_info = PIPELINE_RESULT["models_info"]
    best_model_name = models_info["best_model_name"]
    best_model = models_info["best_model"]
    
    # Get predictions for all test dates
    if best_model_name in ["logreg_pca", "rf_pca"]:
        pred_probs = best_model.predict_proba(X_test_pca)[:, 1]
    else:
        pred_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Build response
    data = []
    for i, date in enumerate(dates_test):
        date_str = date.strftime("%Y-%m-%d")
        spy_price = float(CLOSE_DF.loc[date, "SPY"])
        true_up = int(y_test.iloc[i])
        pred_up_prob = float(pred_probs[i])
        cluster = int(cluster_test[i])
        
        data.append(
            TimeSeriesPoint(
                date=date_str,
                spy_price=spy_price,
                true_up=true_up,
                pred_up_prob=pred_up_prob,
                cluster=cluster,
            )
        )
    
    return TimeSeriesResponse(data=data)


@app.get("/pca_scatter", response_model=PCAScatterResponse)
def get_pca_scatter():
    """
    Returns a sample of the training PCA data for visualization.
    Limited to 2000 samples to keep payload small.
    """
    if PIPELINE_RESULT is None:
        raise ValueError("Pipeline not loaded. Please restart the application.")
    
    X_train_pca = SPLITS["X_train_pca"]
    cluster_train = PIPELINE_RESULT["cluster_train"]
    y_train = SPLITS["y_train"]
    
    # Sample up to 2000 points
    n_samples = min(2000, len(X_train_pca))
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(len(X_train_pca), size=n_samples, replace=False)
    
    data = []
    for idx in indices:
        data.append(
            PCAScatterPoint(
                pc1=float(X_train_pca[idx, 0]),
                pc2=float(X_train_pca[idx, 1]),
                cluster_id=int(cluster_train[idx]),
                target_up=int(y_train.iloc[idx]),
            )
        )
    
    return PCAScatterResponse(data=data)


@app.post("/refresh", response_model=RefreshResponse)
def refresh_pipeline():
    """
    Manually trigger a full pipeline reload.
    - Re-download the full history from Yahoo Finance.
    - Overwrite data/stocks_raw.csv.
    - Re-run the full pipeline and update the in-memory models.
    """
    try:
        logger.info("Manual refresh triggered via /refresh endpoint.")
        _reload_pipeline(reload_data=True)
        return RefreshResponse(status="ok")
    except Exception as e:
        # Make it easy to debug from the frontend or logs
        logger.error(f"Error in manual refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/explain", response_model=LLMExplainResponse)
def llm_explain(payload: LLMExplainRequest):
    """
    Use an LLM to generate an explanation/insight about the current market regime
    and model output.
    - If payload.question is provided, use it as the user question.
    - If no question is provided, generate a default explanation.
    """
    if PIPELINE_RESULT is None or SPLITS is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on the backend")
    
    # Extract key summary info
    models_info = PIPELINE_RESULT["models_info"]
    best_model_name = models_info["best_model_name"]
    cluster_stats = PIPELINE_RESULT["cluster_stats"]
    dates_test = SPLITS["dates_test"]
    latest_date = dates_test[-1].strftime("%Y-%m-%d")
    latest_regime = int(PIPELINE_RESULT["cluster_test"][-1])
    
    # Get latest prediction probability
    X_test_scaled = SPLITS["X_test_scaled"]
    X_test_pca = SPLITS["X_test_pca"]
    latest_idx = len(dates_test) - 1
    
    if best_model_name in ["logreg_pca", "rf_pca"]:
        latest_pred_prob = float(BEST_MODEL.predict_proba(X_test_pca[latest_idx:latest_idx+1])[0, 1])
    else:
        latest_pred_prob = float(BEST_MODEL.predict_proba(X_test_scaled[latest_idx:latest_idx+1])[0, 1])
    
    # Build a compact textual summary of cluster stats
    cluster_lines = []
    for _, row in cluster_stats.iterrows():
        cluster_lines.append(
            f"Cluster {int(row['cluster_id'])}: n_days={int(row['n_days'])}, "
            f"mean_daily_return={row['mean_daily_return']:.4f}, prob_up={row['prob_up']:.3f}"
        )
    cluster_summary_text = "\n".join(cluster_lines)
    
    user_question = payload.question.strip() if payload.question else ""
    if not user_question:
        user_question = (
            "Explain in simple but precise terms what the current market regime means, "
            "how it compares to the other regimes, and how I should interpret the "
            "next-day probability of SPY going up. Avoid financial advice; just "
            "explain the patterns and model output."
        )
    
    system_prompt = (
        "You are an assistant helping a data science student explain a market-regime "
        "detection project. The model looks at multiple ETFs, extracts features, does "
        "PCA + KMeans to find regimes, and then predicts whether SPY will close up "
        "tomorrow. Use clear, technically correct language, but keep it understandable. "
        "Do NOT give investment advice; focus on interpreting regimes and probabilities."
    )
    
    project_context = (
        f"Best model: {best_model_name}\n"
        f"Latest test date: {latest_date}\n"
        f"Current regime (cluster id): {latest_regime}\n"
        f"Next-day probability that SPY closes up: {latest_pred_prob:.3f}\n"
        f"Cluster statistics:\n{cluster_summary_text}\n"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Here is the current model context:\n"
                        f"{project_context}\n\n"
                        f"My question: {user_question}"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    
    return LLMExplainResponse(answer=answer)

