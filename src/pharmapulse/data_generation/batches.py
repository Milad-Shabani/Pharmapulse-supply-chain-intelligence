"""
Generates plant-level production batch records: each SKU is produced
in discrete batches (a real GMP manufacturing pattern - continuous
production is rare in pharma; products are made in campaigns), each
batch carrying its own expiry date derived from the product's shelf
life. This is the data that feeds the expiry-risk / waste-avoidance
part of the planning report - a genuinely pharma-specific concern
that a generic retail-demand project wouldn't need to model.

A small, realistic quality-control failure rate is applied per batch
(QC_pass = False batches are produced but never released to
distribution - they consume plant capacity but don't reach the
network), and batch sizing is loosely tied to the product's average
network-wide daily demand so batch cadence looks plausible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmapulse.data_generation.demand_curve import STUDY_END, STUDY_START


def generate_production_batches(daily_demand: pd.DataFrame, seed: int = 55) -> pd.DataFrame:
    from pharmapulse.data_generation.products import products_dataframe
    from pharmapulse.data_generation.centers_and_plants import centers_dataframe

    rng = np.random.default_rng(seed)
    products = products_dataframe().set_index("product_id")
    centers = centers_dataframe()
    n_centers = centers["center_id"].nunique()

    # Average network-wide daily volume per SKU, used to size batches
    # so that a batch roughly covers 2-4 weeks of network demand.
    avg_daily_network = (
        daily_demand.groupby("product_id")["units_ordered"].mean() * n_centers
    )

    # Production only happens on plant working days (weekdays), and a
    # given SKU is produced in campaigns every ~10-25 days depending on
    # how fast it turns over.
    all_days = pd.date_range(STUDY_START, STUDY_END, freq="D")
    weekdays = all_days[all_days.weekday < 5]

    rows = []
    batch_counter = 0
    for product_id, prod in products.iterrows():
        plant_id = {
            "Analgesics": "PLANT-A", "Antibiotics": "PLANT-B", "Cardiovascular": "PLANT-A",
            "Diabetes": "PLANT-C", "Respiratory": "PLANT-B", "Gastrointestinal": "PLANT-A",
            "Vitamins": "PLANT-C", "Endocrine": "PLANT-B", "Anti-inflammatory": "PLANT-C",
        }.get(prod["category"], "PLANT-A")

        network_daily = max(avg_daily_network.get(product_id, 500), 50)
        cadence_days = int(rng.integers(10, 26))
        batch_size_days_cover = rng.uniform(18, 35)

        day = pd.Timestamp(STUDY_START) - pd.Timedelta(days=int(rng.integers(5, cadence_days)))
        while day <= STUDY_END:
            # snap to nearest following weekday
            while day.weekday() >= 5:
                day += pd.Timedelta(days=1)
            if day > STUDY_END:
                break

            batch_counter += 1
            quantity = int(np.round(network_daily * batch_size_days_cover * rng.uniform(0.85, 1.15)))
            qc_pass = rng.random() > 0.02  # ~2% batch rejection rate, typical of tight GMP control
            expiry_date = day + pd.Timedelta(days=int(prod["shelf_life_days"]))

            rows.append(
                {
                    "batch_id": f"BATCH-{batch_counter:06d}",
                    "product_id": product_id,
                    "plant_id": plant_id,
                    "production_date": day.strftime("%Y-%m-%d"),
                    "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                    "quantity_produced": quantity,
                    "qc_pass": bool(qc_pass),
                }
            )
            day += pd.Timedelta(days=cadence_days + int(rng.integers(-2, 3)))

    return pd.DataFrame(rows)
