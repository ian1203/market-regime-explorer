"""
Core pipeline functions for market regime detection and next-day return prediction.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Tuple, List, Optional

# Suppress harmless joblib resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Configuration constants
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLV"]
DEFAULT_START_DATE = "2010-01-01"


def download_full_history(
    csv_path: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download full daily price history for the configured TICKERS from Yahoo Finance
    and save it as a CSV snapshot at `csv_path`.
    
    Args:
        csv_path: Path where the CSV file will be saved.
        start_date: Start date for historical data (default: "2010-01-01").
        end_date: End date for historical data. If None, uses today's date.
        
    Returns:
        DataFrame with the downloaded data (same structure as expected by load_raw_data).
    """
    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")
    
    # Download data from Yahoo Finance
    print(f"Downloading data from Yahoo Finance for {len(TICKERS)} tickers...")
    print(f"Date range: {start_date} to {end_date}")
    
    df_raw = yf.download(
        tickers=TICKERS,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )
    
    print(f"Downloaded shape: {df_raw.shape}")
    
    # Ensure the directory exists
    csv_path_obj = Path(csv_path)
    csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV (preserving MultiIndex structure)
    df_raw.to_csv(csv_path)
    print(f"Saved CSV snapshot to: {csv_path}")
    
    return df_raw


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Reads stocks_raw.csv into a DataFrame with DateTimeIndex.
    
    Args:
        csv_path: Path to the CSV file containing raw OHLCV data.
        
    Returns:
        DataFrame with DateTimeIndex and MultiIndex columns (field, ticker).
    """
    # Read CSV with MultiIndex columns (header rows 0 and 1)
    df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True, header=[0, 1])
    return df_raw


def build_base_tables(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Extracts close prices, volume, and returns from raw data.
    
    Args:
        df_raw: DataFrame with MultiIndex columns (field, ticker).
        
    Returns:
        Tuple of (close, volume, returns, tickers):
        - close: Adjusted close prices, columns = tickers
        - volume: Volumes, columns = tickers
        - returns: Daily percentage returns per ticker
        - tickers: List of ticker symbols
    """
    if not isinstance(df_raw.columns, pd.MultiIndex):
        raise ValueError(
            "Expected MultiIndex columns from yf.download for multiple tickers. "
            "Inspect df_raw.columns and adjust if needed."
        )
    
    level0 = df_raw.columns.get_level_values(0)
    
    if "Adj Close" in level0:
        price_field = "Adj Close"
    elif "Close" in level0:
        price_field = "Close"
    else:
        raise KeyError(
            "Neither 'Adj Close' nor 'Close' found in the first level of df_raw.columns"
        )
    
    # Extract price and volume per ticker
    close = df_raw[price_field].copy()
    volume = df_raw["Volume"].copy()
    
    # Compute daily percentage returns
    returns = close.pct_change()
    
    # Keep a list of tickers for later steps
    tickers = list(close.columns)
    
    return close, volume, returns, tickers


def build_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Build a high-dimensional feature table from prices, volume and returns.
    Features are computed per ticker and concatenated for all tickers.
    
    Args:
        close: Adjusted close prices, columns = tickers
        volume: Volumes, columns = tickers
        returns: Daily percentage returns per ticker
        tickers: List of ticker symbols
        
    Returns:
        DataFrame indexed by date, columns = all engineered feature names.
    """
    feats = pd.DataFrame(index=close.index)
    
    lag_list = [1, 2, 5, 10]
    mean_windows = [5, 10, 20, 50]
    vol_windows = [5, 10, 20, 50]
    vol_ma_windows = [5, 20]
    
    for ticker in tickers:
        r = returns[ticker]
        v = volume[ticker]
        
        # Lagged returns
        for lag in lag_list:
            feats[f"{ticker}_ret_lag{lag}"] = r.shift(lag)
        
        # Rolling mean of returns
        for win in mean_windows:
            feats[f"{ticker}_ret_mean_{win}"] = r.rolling(win).mean()
        
        # Rolling volatility (std of returns)
        for win in vol_windows:
            feats[f"{ticker}_ret_std_{win}"] = r.rolling(win).std()
        
        # Volume moving averages
        for win in vol_ma_windows:
            feats[f"{ticker}_vol_ma_{win}"] = v.rolling(win).mean()
        
        # Volume relative to 20-day moving average
        vol_ma_20 = v.rolling(20).mean()
        feats[f"{ticker}_vol_rel_20"] = v / vol_ma_20
    
    return feats


def build_dataset(features: pd.DataFrame, close: pd.DataFrame) -> Dict:
    """
    Creates the target series, applies time-based split, and standardizes features.
    
    Args:
        features: DataFrame with engineered features
        close: DataFrame with close prices (to extract SPY for target)
        
    Returns:
        Dictionary with:
        - X_train_scaled, X_val_scaled, X_test_scaled (numpy arrays)
        - X_train_pca, X_val_pca, X_test_pca (numpy arrays)
        - y_train, y_val, y_test (Series)
        - scaler, pca (fitted transformers)
        - dates_train, dates_val, dates_test (Index)
    """
    # Compute SPY daily returns and target
    spy_price = close["SPY"]
    spy_returns = spy_price.pct_change()
    
    # Binary target: 1 if next-day return > 0, else 0
    target_up = (spy_returns.shift(-1) > 0).astype(int)
    
    # Combine features and target, then drop NaNs
    data = features.copy()
    data["target_up"] = target_up
    data = data.dropna()
    
    # Separate features X and target y
    X = data.drop(columns=["target_up"])
    y = data["target_up"]
    
    # Time-based split
    dates = data.index
    train_end = "2018-12-31"
    val_end = "2021-12-31"
    
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    dates_train = X_train.index
    dates_val = X_val.index
    dates_test = X_test.index
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA with 20 components
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,
        "X_val_pca": X_val_pca,
        "X_test_pca": X_test_pca,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "pca": pca,
        "dates_train": dates_train,
        "dates_val": dates_val,
        "dates_test": dates_test,
    }


def fit_kmeans(X_train_pca: np.ndarray, n_clusters: int = 2) -> KMeans:
    """
    Fits KMeans on the PCA train data.
    
    Args:
        X_train_pca: PCA-transformed training features
        n_clusters: Number of clusters (default: 2)
        
    Returns:
        Fitted KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X_train_pca)
    return kmeans


def assign_clusters(
    kmeans: KMeans,
    X_train_pca: np.ndarray,
    X_val_pca: np.ndarray,
    X_test_pca: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Assigns cluster labels to train, validation, and test sets.
    
    Args:
        kmeans: Fitted KMeans model
        X_train_pca: PCA-transformed training features
        X_val_pca: PCA-transformed validation features
        X_test_pca: PCA-transformed test features
        
    Returns:
        Dictionary with cluster_train, cluster_val, cluster_test
    """
    cluster_train = kmeans.predict(X_train_pca)
    cluster_val = kmeans.predict(X_val_pca)
    cluster_test = kmeans.predict(X_test_pca)
    
    return {
        "cluster_train": cluster_train,
        "cluster_val": cluster_val,
        "cluster_test": cluster_test,
    }


def train_models(splits: Dict) -> Dict:
    """
    Trains four models and evaluates them on the test set.
    
    Args:
        splits: Dictionary from build_dataset() containing train/val/test splits
        
    Returns:
        Dictionary with:
        - models: dict of trained models
        - metrics: DataFrame with model performance metrics
        - best_model_name: name of the best model
        - best_model: the corresponding model instance
    """
    X_train_scaled = splits["X_train_scaled"]
    X_val_scaled = splits["X_val_scaled"]
    X_test_scaled = splits["X_test_scaled"]
    X_train_pca = splits["X_train_pca"]
    X_val_pca = splits["X_val_pca"]
    X_test_pca = splits["X_test_pca"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]
    
    models = {}
    results = []
    
    # Logistic Regression on full features
    logreg_full = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    logreg_full.fit(X_train_scaled, y_train)
    y_pred = logreg_full.predict(X_test_scaled)
    y_proba = logreg_full.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    models["logreg_full"] = logreg_full
    results.append({
        "model": "logreg_full",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
    })
    
    # Logistic Regression on PCA features
    logreg_pca = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    logreg_pca.fit(X_train_pca, y_train)
    y_pred = logreg_pca.predict(X_test_pca)
    y_proba = logreg_pca.predict_proba(X_test_pca)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    models["logreg_pca"] = logreg_pca
    results.append({
        "model": "logreg_pca",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
    })
    
    # Random Forest on full features
    rf_full = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_full.fit(X_train_scaled, y_train)
    y_pred = rf_full.predict(X_test_scaled)
    y_proba = rf_full.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    models["rf_full"] = rf_full
    results.append({
        "model": "rf_full",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
    })
    
    # Random Forest on PCA features
    rf_pca = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_pca.fit(X_train_pca, y_train)
    y_pred = rf_pca.predict(X_test_pca)
    y_proba = rf_pca.predict_proba(X_test_pca)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    models["rf_pca"] = rf_pca
    results.append({
        "model": "rf_pca",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
    })
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Find best model (by ROC-AUC, then F1)
    best_idx = metrics_df.sort_values(["roc_auc", "f1"], ascending=False).index[0]
    best_model_name = metrics_df.loc[best_idx, "model"]
    best_model = models[best_model_name]
    
    return {
        "models": models,
        "metrics": metrics_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }


def compute_cluster_stats(
    cluster_train: np.ndarray,
    y_train: pd.Series,
    spy_returns_train: pd.Series
) -> pd.DataFrame:
    """
    Computes statistics for each cluster.
    
    Args:
        cluster_train: Cluster labels for training data (numpy array)
        y_train: Target labels (0/1) for training data (pandas Series)
        spy_returns_train: SPY daily returns for training data (pandas Series, aligned with y_train)
        
    Returns:
        DataFrame with columns: cluster_id, n_days, mean_daily_return, prob_up
    """
    stats = []
    
    # Ensure spy_returns_train is aligned with y_train (same index)
    if not spy_returns_train.index.equals(y_train.index):
        spy_returns_train = spy_returns_train.reindex(y_train.index)
    
    unique_clusters = np.unique(cluster_train)
    for cluster_id in unique_clusters:
        mask = cluster_train == cluster_id
        n_days = mask.sum()
        # Use iloc for positional indexing to ensure alignment
        mean_return = spy_returns_train.iloc[mask].mean()
        prob_up = y_train.iloc[mask].mean()
        
        stats.append({
            "cluster_id": int(cluster_id),
            "n_days": int(n_days),
            "mean_daily_return": float(mean_return),
            "prob_up": float(prob_up),
        })
    
    return pd.DataFrame(stats)


def run_full_pipeline(csv_path: str) -> Dict:
    """
    Convenience function that runs the entire pipeline.
    
    Args:
        csv_path: Path to stocks_raw.csv
        
    Returns:
        Dictionary with all pipeline results:
        - splits: train/val/test splits
        - kmeans: fitted KMeans model
        - cluster_train, cluster_val, cluster_test: cluster assignments
        - models_info: model training results
        - cluster_stats: cluster statistics
        - close: close prices DataFrame
        - features: features DataFrame
    """
    # Load raw data
    df_raw = load_raw_data(csv_path)
    
    # Build base tables
    close, volume, returns, tickers = build_base_tables(df_raw)
    
    # Build features
    features = build_features(close, volume, returns, tickers)
    
    # Build dataset and splits
    splits = build_dataset(features, close)
    
    # Fit KMeans
    kmeans = fit_kmeans(splits["X_train_pca"], n_clusters=2)
    
    # Assign clusters
    clusters = assign_clusters(
        kmeans,
        splits["X_train_pca"],
        splits["X_val_pca"],
        splits["X_test_pca"],
    )
    
    # Train models
    models_info = train_models(splits)
    
    # Compute cluster stats (need SPY returns for training period)
    spy_returns_train = close["SPY"].pct_change().loc[splits["dates_train"]]
    cluster_stats = compute_cluster_stats(
        clusters["cluster_train"],
        splits["y_train"],
        spy_returns_train,
    )
    
    return {
        "splits": splits,
        "kmeans": kmeans,
        "cluster_train": clusters["cluster_train"],
        "cluster_val": clusters["cluster_val"],
        "cluster_test": clusters["cluster_test"],
        "models_info": models_info,
        "cluster_stats": cluster_stats,
        "close": close,
        "features": features,
    }

