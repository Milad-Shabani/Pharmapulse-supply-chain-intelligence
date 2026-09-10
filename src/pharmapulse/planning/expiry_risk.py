"""
Pharma-specific waste-risk analysis: which products are carrying stock
that is likely to expire before it can be sold, given current network
inventory and forecasted future demand.

Methodology (documented approximation - batch-level, per-unit lot
tracking through the distribution network is out of scope for this
project, but the FIFO assumption below is exactly the real-world
policy pharma distributors are required to follow for expiry
management, so it is a reasonable proxy):

1. For each product, take all QC-passed production batches that have
   not yet expired as of the "as-of" date (the last day of the
   observed dataset).
2. Sort those batches by expiry date, ascending (FIFO: the
   soonest-to-expire stock is the oldest production, and - because
   FIFO put it into circulation first - is assumed to be what's
   currently sitting in the network).
3. Accumulate batch quantities, oldest-expiry-first, until the running
   total reaches the product's current total network stock (from the
   inventory ledger). The batches consumed in this walk are treated as
   "what's currently on the shelf", and their expiry dates are used to
   bucket at-risk quantity into 30/60/90-day windows.
4. Compare each window's at-risk quantity to the forecasted demand
   for that same window - if forecasted demand can't consume the
   at-risk quantity in time, it is flagged as a waste risk.
"""
from __future__ import annotations

import pandas as pd


def compute_expiry_risk(
    batches: pd.DataFrame,
    current_stock_by_product: pd.Series,  # index: product_id, value: total network units on hand
    forecast_weekly_by_product: pd.DataFrame,  # columns: product_id, week_start, forecast_p50
    as_of_date: pd.Timestamp,
    products: pd.DataFrame,
) -> pd.DataFrame:
    passed = batches[batches["qc_pass"]].copy()
    passed["expiry_date"] = pd.to_datetime(passed["expiry_date"])
    passed["production_date"] = pd.to_datetime(passed["production_date"])
    unexpired = passed[passed["expiry_date"] >= as_of_date].sort_values(["product_id", "expiry_date"])

    rows = []
    for product_id, group in unexpired.groupby("product_id"):
        stock_remaining = float(current_stock_by_product.get(product_id, 0))
        if stock_remaining <= 0:
            continue

        running = 0.0
        exp_30 = exp_60 = exp_90 = 0.0
        for batch in group.itertuples(index=False):
            if running >= stock_remaining:
                break
            take = min(float(batch.quantity_produced), stock_remaining - running)
            running += take
            days_to_expiry = (batch.expiry_date - as_of_date).days
            if days_to_expiry <= 30:
                exp_30 += take
            if days_to_expiry <= 60:
                exp_60 += take
            if days_to_expiry <= 90:
                exp_90 += take

        # Forecasted demand over the same forward windows, network-wide.
        fc = forecast_weekly_by_product[forecast_weekly_by_product["product_id"] == product_id]
        fc = fc.sort_values("week_start")
        demand_30 = fc.iloc[:4]["forecast_p50"].sum() if len(fc) else 0.0
        demand_60 = fc.iloc[:8]["forecast_p50"].sum() if len(fc) else 0.0
        demand_90 = fc.iloc[:12]["forecast_p50"].sum() if len(fc) else 0.0

        rows.append(
            {
                "product_id": product_id,
                "current_network_stock": round(stock_remaining),
                "units_expiring_30d": round(exp_30),
                "units_expiring_60d": round(exp_60),
                "units_expiring_90d": round(exp_90),
                "forecasted_demand_30d": round(demand_30),
                "forecasted_demand_60d": round(demand_60),
                "forecasted_demand_90d": round(demand_90),
                "at_risk_units_30d": max(round(exp_30 - demand_30), 0),
                "waste_risk_flag": exp_30 > demand_30 * 1.1,  # 10% buffer before flagging
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.merge(products[["product_id", "generic_name", "category", "unit_cost"]], on="product_id")
        result["at_risk_value"] = (result["at_risk_units_30d"] * result["unit_cost"]).round(2)
        result = result.sort_values("at_risk_value", ascending=False).reset_index(drop=True)
    return result
