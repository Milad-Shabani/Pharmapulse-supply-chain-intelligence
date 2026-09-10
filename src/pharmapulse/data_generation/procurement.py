"""
Simulates the upstream side of the supply chain: raw-material
consumption driven by the production batch schedule (via the BOM),
purchase orders placed against suppliers with their own lead times
and on-time delivery reliability, and the resulting raw-material
inventory position over time.

This is what makes "supply risk to production" a first-class,
measurable thing in this dataset rather than an assumption: an
overseas API supplier with an 80% on-time rate and a 55-day lead time
genuinely can (and, in a handful of simulated instances, does) leave a
material's on-hand stock too low to safely cover the next production
campaign - exactly the kind of signal a procurement/S&OP dashboard
exists to surface before it becomes a missed batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_material_consumption(batches: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    merged = batches[batches["qc_pass"]].merge(bom, on="product_id")
    merged["qty_consumed"] = merged["quantity_produced"] / 1000.0 * merged["qty_per_1000_units"]
    consumption = merged.groupby(["material_id", "production_date"], as_index=False)["qty_consumed"].sum()
    consumption = consumption.rename(columns={"production_date": "date"})
    return consumption


def simulate_procurement(
    consumption: pd.DataFrame,
    materials: pd.DataFrame,
    study_start: pd.Timestamp,
    study_end: pd.Timestamp,
    review_period_days: int = 7,
    seed: int = 88,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    dates = pd.date_range(study_start, study_end, freq="D")
    n_days = len(dates)
    material_ids = materials["material_id"].tolist()
    n_materials = len(material_ids)
    mat_index = {m: i for i, m in enumerate(material_ids)}

    daily_consumption = np.zeros((n_days, n_materials))
    date_index = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    for row in consumption.itertuples(index=False):
        di = date_index.get(row.date)
        mi = mat_index.get(row.material_id)
        if di is not None and mi is not None:
            daily_consumption[di, mi] += row.qty_consumed

    lead_times = materials.set_index("material_id").loc[material_ids, "avg_lead_time_days"].to_numpy()
    on_time_rates = materials.set_index("material_id").loc[material_ids, "on_time_rate"].to_numpy()
    unit_costs = materials.set_index("material_id").loc[material_ids, "unit_cost"].to_numpy()

    warmup = daily_consumption[:90].mean(axis=0)
    full_avg = daily_consumption.mean(axis=0)
    avg_daily = np.where(warmup > 0, warmup, full_avg)
    avg_daily = np.maximum(avg_daily, 0.01)

    reorder_point = avg_daily * lead_times * 1.5
    order_up_to = reorder_point + avg_daily * (lead_times + review_period_days)

    max_lead = int(lead_times.max()) + int(lead_times.max() * 0.4) + 5
    pending = np.zeros((max_lead + 1, n_materials))
    on_hand = order_up_to.copy()
    on_order = np.zeros(n_materials)

    stock_history = np.zeros((n_days, n_materials))
    po_records = []
    po_counter = 0
    supplier_lookup = materials.set_index("material_id")["supplier_id"]

    for t in range(n_days):
        idx = t % (max_lead + 1)
        arrivals = pending[idx].copy()
        pending[idx] = 0.0
        on_hand += arrivals
        on_order -= arrivals

        on_hand -= daily_consumption[t]
        on_hand = np.maximum(on_hand, 0)

        if t % review_period_days == 0:
            position = on_hand + on_order
            need_reorder = position < reorder_point
            for mi in np.nonzero(need_reorder)[0]:
                qty = max(order_up_to[mi] - position[mi], avg_daily[mi] * review_period_days)
                material_id = material_ids[mi]
                is_on_time = rng.random() < on_time_rates[mi]
                delay = 0 if is_on_time else int(rng.integers(4, 21))
                actual_lead = int(lead_times[mi]) + delay
                arrival_day = min(t + actual_lead, n_days - 1)
                arrival_slot = arrival_day % (max_lead + 1)

                po_counter += 1
                order_date = dates[t]
                expected_delivery = order_date + pd.Timedelta(days=int(lead_times[mi]))
                actual_delivery = dates[arrival_day]
                po_records.append({
                    "po_id": f"PO-{po_counter:06d}",
                    "material_id": material_id,
                    "supplier_id": supplier_lookup.loc[material_id],
                    "order_date": order_date.strftime("%Y-%m-%d"),
                    "expected_delivery_date": expected_delivery.strftime("%Y-%m-%d"),
                    "actual_delivery_date": actual_delivery.strftime("%Y-%m-%d"),
                    "quantity_ordered": round(qty, 1),
                    "unit_cost": unit_costs[mi],
                    "on_time": bool(is_on_time),
                    "delay_days": delay,
                })
                pending[arrival_slot, mi] += qty
                on_order[mi] += qty

        stock_history[t] = on_hand

    inventory_df = pd.DataFrame(stock_history, columns=material_ids)
    inventory_df.insert(0, "date", dates.strftime("%Y-%m-%d"))
    inventory_long = inventory_df.melt(id_vars="date", var_name="material_id", value_name="closing_stock")

    consumption_wide = pd.DataFrame(daily_consumption, columns=material_ids)
    consumption_wide.insert(0, "date", dates.strftime("%Y-%m-%d"))
    consumption_long = consumption_wide.melt(id_vars="date", var_name="material_id", value_name="qty_consumed")

    inventory_long = inventory_long.merge(consumption_long, on=["date", "material_id"])

    purchase_orders = pd.DataFrame(po_records)
    return purchase_orders, inventory_long
