import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pharmapulse.data_generation.products import products_dataframe
from pharmapulse.data_generation.raw_materials import build_supply_network
from pharmapulse.data_generation.demand_curve import generate_daily_demand, STUDY_START, STUDY_END
from pharmapulse.data_generation.batches import generate_production_batches
from pharmapulse.data_generation.procurement import compute_material_consumption, simulate_procurement


def test_every_product_has_at_least_one_bom_line():
    products = products_dataframe()
    net = build_supply_network(products)
    covered = set(net["bom"]["product_id"].unique())
    assert covered == set(products["product_id"])


def test_bom_materials_exist_in_materials_table():
    products = products_dataframe()
    net = build_supply_network(products)
    assert set(net["bom"]["material_id"]).issubset(set(net["materials"]["material_id"]))


def test_materials_map_to_valid_suppliers():
    products = products_dataframe()
    net = build_supply_network(products)
    assert set(net["materials"]["supplier_id"]).issubset(set(net["suppliers"]["supplier_id"]))


def test_purchase_orders_have_sane_delivery_dates():
    products = products_dataframe()
    net = build_supply_network(products)
    demand = generate_daily_demand(seed=2)
    batches = generate_production_batches(demand, seed=2)
    consumption = compute_material_consumption(batches, net["bom"])
    po, _ = simulate_procurement(consumption, net["materials"], STUDY_START, STUDY_END)

    order_date = pd.to_datetime(po["order_date"])
    actual = pd.to_datetime(po["actual_delivery_date"])
    assert (actual >= order_date).all()
    assert (po["quantity_ordered"] > 0).all()


def test_on_time_orders_have_zero_delay():
    products = products_dataframe()
    net = build_supply_network(products)
    demand = generate_daily_demand(seed=2)
    batches = generate_production_batches(demand, seed=2)
    consumption = compute_material_consumption(batches, net["bom"])
    po, _ = simulate_procurement(consumption, net["materials"], STUDY_START, STUDY_END)

    assert (po.loc[po["on_time"], "delay_days"] == 0).all()
    assert (po.loc[~po["on_time"], "delay_days"] > 0).all()
