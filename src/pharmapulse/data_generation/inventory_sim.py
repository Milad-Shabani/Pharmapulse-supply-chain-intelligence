"""
Simulates daily on-hand inventory for every (product, distribution
center) pair, driven by the generated demand series and a classic
periodic-review (s, S) replenishment policy - reorder up to S whenever
the inventory position (on-hand + on-order) drops below the reorder
point s, with the order arriving `lead_time_days` later.

This produces `stockout_units` (unmet demand on days inventory ran
out) and `closing_stock`, which are exactly the operational pain
points a demand-planning project is meant to reduce - and which the
forecasting + safety-stock layer later in the pipeline is evaluated
against.

Note: the reorder point/order-up-to level here are computed from each
series' own full-history average demand (a simplification appropriate
for *generating a plausible operational history*; the actual planning
recommendations produced later in the pipeline are the point of the
project and are computed properly from held-out forecasts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_inventory(
    daily_demand: pd.DataFrame,
    centers: pd.DataFrame,
    review_period_days: int = 7,
    service_z: float = 1.65,  # ~95% cycle service level
    seed: int = 77,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    wide = daily_demand.pivot_table(
        index="date", columns=["product_id", "center_id"], values="units_ordered", fill_value=0
    ).sort_index()
    dates = pd.to_datetime(wide.index)
    n_days, n_series = wide.shape
    demand = wide.to_numpy(dtype=float)

    columns = wide.columns  # MultiIndex (product_id, center_id)
    center_lead_time = centers.set_index("center_id")["lead_time_days_from_plant"]
    lead_times = np.array([center_lead_time[c] for (_, c) in columns], dtype=int)

    mean_d = demand.mean(axis=0)
    std_d = demand.std(axis=0)

    reorder_point = mean_d * lead_times + service_z * std_d * np.sqrt(np.maximum(lead_times, 1))
    order_up_to = reorder_point + mean_d * review_period_days

    max_lead = int(lead_times.max()) + 1
    pending = np.zeros((max_lead + 1, n_series))

    on_hand = np.maximum(order_up_to * rng.uniform(0.8, 1.1, size=n_series), mean_d * 3)
    on_order = np.zeros(n_series)

    records_stock = np.zeros((n_days, n_series))
    records_stockout = np.zeros((n_days, n_series))
    records_received = np.zeros((n_days, n_series))
    records_ordered = np.zeros((n_days, n_series))

    for t in range(n_days):
        idx = t % (max_lead + 1)
        arrivals = pending[idx].copy()
        pending[idx] = 0.0
        on_hand += arrivals
        on_order -= arrivals
        records_received[t] = arrivals

        sold = np.minimum(on_hand, demand[t])
        stockout = demand[t] - sold
        on_hand -= sold
        records_stockout[t] = stockout

        if t % review_period_days == 0:
            position = on_hand + on_order
            need_reorder = position < reorder_point
            order_qty = np.where(need_reorder, np.maximum(order_up_to - position, 0), 0.0)
            nonzero = np.nonzero(order_qty > 0)[0]
            for series_i in nonzero:
                arrival_slot = (t + lead_times[series_i]) % (max_lead + 1)
                pending[arrival_slot, series_i] += order_qty[series_i]
            on_order += order_qty
            records_ordered[t] = order_qty

        records_stock[t] = on_hand

    out_frames = []
    date_strs = dates.strftime("%Y-%m-%d")
    for i, (product_id, center_id) in enumerate(columns):
        units_sold = demand[:, i] - records_stockout[:, i]
        out_frames.append(
            pd.DataFrame(
                {
                    "date": date_strs,
                    "product_id": product_id,
                    "center_id": center_id,
                    "units_demanded": demand[:, i].astype(int),
                    "units_sold": np.round(units_sold).astype(int),
                    "closing_stock": np.round(records_stock[:, i]).astype(int),
                    "stockout_units": np.round(records_stockout[:, i]).astype(int),
                    "units_received": np.round(records_received[:, i]).astype(int),
                    "units_reordered": np.round(records_ordered[:, i]).astype(int),
                }
            )
        )
    return pd.concat(out_frames, ignore_index=True)
