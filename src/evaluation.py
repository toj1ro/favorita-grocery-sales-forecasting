import numpy as np
import pandas as pd
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    y_pred = np.clip(y_pred, 0, None)
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    if weights is not None:
        return float(np.sqrt(np.mean(weights * log_diff ** 2)))
    return float(np.sqrt(np.mean(log_diff ** 2)))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        weights: np.ndarray | None = None) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred, weights),
    }


def evaluate_per_series(df: pd.DataFrame, y_col: str, pred_col: str,
                        weight_col: str | None = None) -> pd.DataFrame:
    results = []
    for uid, group in df.groupby("unique_id"):
        y = group[y_col].values
        p = group[pred_col].values
        w = group[weight_col].values if weight_col else None
        metrics = compute_all_metrics(y, p, w)
        metrics["unique_id"] = uid
        metrics["n_points"] = len(y)
        results.append(metrics)
    return pd.DataFrame(results)


def summary_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.round(4)
