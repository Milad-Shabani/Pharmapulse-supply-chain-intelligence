"""
Converts the P50/P95 weekly demand forecasts into operational
inventory recommendations: reorder point (ROP) and order-up-to level
(OUL), using each series' own forecast *quantile spread* as the
safety-stock buffer rather than assuming demand is Normally
distributed (the classical `mean + z * sigma` formula). This matters
here because several SKU categories (antibiotics, allergy meds) have
sharply right-skewed, seasonal demand that a Normal-distribution
assumption underestimates.

    daily_p50 = weekly_p50_forecast / 7
    daily_p95 = weekly_p95_forecast / 7
    expected_demand_over_lead_time = daily_p50 * lead_time_days
    safety_stock = (daily_p95 - daily_p50) * lead_time_days
    reorder_point = expected_demand_over_lead_time + safety_stock
    order_up_to   = reorder_point + daily_p50 * review_period_days
"""
from __future__ import annotations

import pandas as pd


def compute_safety_stock(
    forecast_df: pd.DataFrame,
    centers: pd.DataFrame,
    review_period_days: int = 7,
) -> pd.DataFrame:
    df = forecast_df.merge(
        centers[["center_id", "lead_time_days_from_plant"]], on="center_id", how="left"
    )
    lead_time = df["lead_time_days_from_plant"]

    df["daily_forecast_p50"] = df["forecast_p50"] / 7
    df["daily_forecast_p95"] = df["forecast_p95"] / 7
    df["expected_demand_over_lead_time"] = df["daily_forecast_p50"] * lead_time
    df["safety_stock_units"] = (
        (df["daily_forecast_p95"] - df["daily_forecast_p50"]) * lead_time
    ).clip(lower=0)
    df["reorder_point_units"] = df["expected_demand_over_lead_time"] + df["safety_stock_units"]
    df["order_up_to_units"] = df["reorder_point_units"] + df["daily_forecast_p50"] * review_period_days

    return df.round(
        {
            "daily_forecast_p50": 1, "daily_forecast_p95": 1,
            "expected_demand_over_lead_time": 0, "safety_stock_units": 0,
            "reorder_point_units": 0, "order_up_to_units": 0,
        }
    )
