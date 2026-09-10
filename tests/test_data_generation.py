import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pharmapulse.data_generation.products import products_dataframe
from pharmapulse.data_generation.centers_and_plants import centers_dataframe, plants_dataframe
from pharmapulse.data_generation.demand_curve import generate_daily_demand, STUDY_START, STUDY_END
from pharmapulse.data_generation.batches import generate_production_batches


def test_product_ids_unique():
    df = products_dataframe()
    assert df["product_id"].is_unique
    assert len(df) >= 40


def test_products_have_positive_economics():
    df = products_dataframe()
    assert (df["unit_price"] > df["unit_cost"]).all()
    assert (df["shelf_life_days"] > 0).all()


def test_center_ids_unique_and_map_to_valid_plants():
    centers = centers_dataframe()
    plants = plants_dataframe()
    assert centers["center_id"].is_unique
    assert centers["supplying_plant_id"].isin(plants["plant_id"]).all()


def test_daily_demand_shape_and_bounds():
    demand = generate_daily_demand(seed=1)
    n_products = products_dataframe().shape[0]
    n_centers = centers_dataframe().shape[0]
    n_days = (STUDY_END - STUDY_START).days + 1
    assert len(demand) == n_products * n_centers * n_days
    assert (demand["units_ordered"] >= 0).all()


def test_daily_demand_reproducible_with_seed():
    d1 = generate_daily_demand(seed=5)
    d2 = generate_daily_demand(seed=5)
    pd.testing.assert_frame_equal(d1, d2)


def test_chronic_category_has_less_seasonal_swing_than_antibiotics():
    """Cardiovascular (flat_chronic) should show much less relative
    week-to-week swing than Antibiotics (strong_winter)."""
    demand = generate_daily_demand(seed=1)
    products = products_dataframe()
    merged = demand.merge(products[["product_id", "category"]], on="product_id")
    merged["date"] = pd.to_datetime(merged["date"])

    def relative_swing(category: str) -> float:
        sub = merged[merged["category"] == category].groupby("date")["units_ordered"].sum()
        return sub.std() / sub.mean()

    assert relative_swing("Antibiotics") > relative_swing("Cardiovascular")


def test_production_batches_expiry_after_production():
    demand = generate_daily_demand(seed=1)
    batches = generate_production_batches(demand, seed=1)
    prod = pd.to_datetime(batches["production_date"])
    exp = pd.to_datetime(batches["expiry_date"])
    assert (exp > prod).all()
    assert batches["quantity_produced"].min() > 0
