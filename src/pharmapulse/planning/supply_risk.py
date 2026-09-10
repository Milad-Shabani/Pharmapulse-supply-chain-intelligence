"""
Upstream supply-chain risk metrics: supplier on-time delivery
performance, and raw-material "days of cover" - how many days
current stock would last at the recent consumption rate, compared
against that material's own supplier lead time. A material whose
days-of-cover has fallen below its lead time is a genuine production
risk: if it stocked out today, the replenishment wouldn't arrive in
time to avoid delaying the next production batch.
"""
from __future__ import annotations

import pandas as pd


def compute_supplier_performance(purchase_orders: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    perf = purchase_orders.groupby("supplier_id").agg(
        orders_placed=("po_id", "count"),
        on_time_rate=("on_time", "mean"),
        avg_delay_days=("delay_days", "mean"),
        total_spend=("quantity_ordered", lambda q: float((q * purchase_orders.loc[q.index, "unit_cost"]).sum())),
    ).reset_index()
    perf = perf.merge(suppliers[["supplier_id", "supplier_name", "region", "category"]], on="supplier_id")
    return perf.sort_values("on_time_rate")


def compute_material_coverage(
    raw_material_inventory: pd.DataFrame,
    materials: pd.DataFrame,
    as_of_date: str,
    trailing_window_days: int = 60,
) -> pd.DataFrame:
    df = raw_material_inventory.copy()
    df["date"] = pd.to_datetime(df["date"])
    as_of = pd.to_datetime(as_of_date)
    window_start = as_of - pd.Timedelta(days=trailing_window_days)

    latest_stock = df[df["date"] == as_of].set_index("material_id")["closing_stock"]
    trailing = df[(df["date"] > window_start) & (df["date"] <= as_of)]
    avg_daily_consumption = trailing.groupby("material_id")["qty_consumed"].mean()

    coverage = pd.DataFrame({
        "current_stock": latest_stock,
        "avg_daily_consumption": avg_daily_consumption,
    }).reset_index()
    coverage["avg_daily_consumption"] = coverage["avg_daily_consumption"].clip(lower=0.01)
    coverage["days_of_cover"] = coverage["current_stock"] / coverage["avg_daily_consumption"]

    coverage = coverage.merge(
        materials[["material_id", "material_name", "material_category", "supplier_id", "avg_lead_time_days"]],
        on="material_id",
    )
    coverage["at_risk"] = coverage["days_of_cover"] < coverage["avg_lead_time_days"]
    return coverage.sort_values("days_of_cover")
