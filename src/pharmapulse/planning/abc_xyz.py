"""
Classic ABC/XYZ inventory segmentation, computed at (product, center)
grain so that inventory policy can be differentiated down to the
level replenishment decisions are actually made.

  * ABC - Pareto classification by revenue contribution.
      A: cumulative revenue share up to 80%
      B: next up to 95%
      C: remaining long tail
  * XYZ - classification by demand variability (coefficient of
    variation of weekly demand).
      X: CV < 0.5  (stable, easy to forecast)
      Y: 0.5 <= CV < 1.0 (moderate variability)
      Z: CV >= 1.0 (erratic, hard to forecast - typically acute-care /
         promotion-driven SKUs)

The combined 9-cell ABC-XYZ matrix is the standard lens demand
planners use to decide *how much planning effort* a given SKU-center
deserves: AX gets tight, forecast-driven control; CZ often gets a
simple, generously-buffered min/max policy because the cost of
sophisticated forecasting isn't justified by the revenue at stake.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_abc_xyz(daily_demand: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = daily_demand.merge(products[["product_id", "unit_price"]], on="product_id")
    df["revenue"] = df["units_ordered"] * df["unit_price"]

    grp = df.groupby(["product_id", "center_id"])
    summary = grp.agg(
        total_revenue=("revenue", "sum"),
        total_units=("units_ordered", "sum"),
    ).reset_index()

    weekly = df.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly["week_start"] = weekly["date"] - pd.to_timedelta(weekly["date"].dt.weekday, unit="D")
    weekly_units = weekly.groupby(["product_id", "center_id", "week_start"])["units_ordered"].sum()
    cv = weekly_units.groupby(["product_id", "center_id"]).agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)
    summary = summary.merge(cv.rename("demand_cv").reset_index(), on=["product_id", "center_id"])

    summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    summary["revenue_share"] = summary["total_revenue"] / summary["total_revenue"].sum()
    summary["cum_revenue_share"] = summary["revenue_share"].cumsum()

    summary["abc_class"] = np.select(
        [summary["cum_revenue_share"] <= 0.80, summary["cum_revenue_share"] <= 0.95],
        ["A", "B"],
        default="C",
    )
    summary["xyz_class"] = pd.cut(
        summary["demand_cv"], bins=[-np.inf, 0.5, 1.0, np.inf], labels=["X", "Y", "Z"]
    )
    summary["segment"] = summary["abc_class"] + summary["xyz_class"].astype(str)

    return summary
