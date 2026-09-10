"""Common forecast-accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 5.0) -> float:
    denom = np.clip(np.abs(y_true), epsilon, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error - the standard metric for demand
    planning at the aggregate/portfolio level (robust to individual
    low-volume series with near-zero actuals, unlike plain MAPE)."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def bias_pct(y_true, y_pred) -> float:
    """Signed forecast bias - positive means the model systematically
    over-forecasts, negative means it systematically under-forecasts."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(y_pred - y_true) / denom * 100)


def r_squared(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def summarize(y_true, y_pred, label: str) -> dict:
    return {
        "segment": label,
        "MAE": round(mae(y_true, y_pred), 1),
        "RMSE": round(rmse(y_true, y_pred), 1),
        "WAPE_%": round(wape(y_true, y_pred), 2),
        "MAPE_%": round(mape(y_true, y_pred), 2),
        "Bias_%": round(bias_pct(y_true, y_pred), 2),
        "R2": round(r_squared(y_true, y_pred), 4),
        "n": len(y_true),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("segment")
