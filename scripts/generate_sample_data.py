"""
Generates the full synthetic raw dataset for PharmaPulse: product
catalog, network topology, ~430K-row daily demand fact table,
production batch/expiry records, and the simulated inventory ledger.

Usage:
    python scripts/generate_sample_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pharmapulse.data_generation.products import products_dataframe  # noqa: E402
from pharmapulse.data_generation.centers_and_plants import (  # noqa: E402
    centers_dataframe, plants_dataframe,
)
from pharmapulse.data_generation.demand_curve import (  # noqa: E402
    generate_daily_demand, STUDY_START, STUDY_END,
)
from pharmapulse.data_generation.batches import generate_production_batches  # noqa: E402
from pharmapulse.data_generation.inventory_sim import simulate_inventory  # noqa: E402
from pharmapulse.data_generation.raw_materials import build_supply_network  # noqa: E402
from pharmapulse.data_generation.procurement import (  # noqa: E402
    compute_material_consumption, simulate_procurement,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Generating product catalog...")
    products = products_dataframe()
    products.to_csv(RAW_DIR / "products.csv", index=False)
    print(f"  -> {len(products)} SKUs across {products['category'].nunique()} categories")

    print("Generating network topology (plants + distribution centers)...")
    plants = plants_dataframe()
    centers = centers_dataframe()
    plants.to_csv(RAW_DIR / "manufacturing_plants.csv", index=False)
    centers.to_csv(RAW_DIR / "distribution_centers.csv", index=False)
    print(f"  -> {len(plants)} plants, {len(centers)} distribution centers")

    print("Generating daily demand fact table (this is the large one)...")
    demand = generate_daily_demand()
    demand.to_parquet(RAW_DIR / "daily_demand.parquet", index=False)
    demand.head(5000).to_csv(RAW_DIR / "daily_demand_sample.csv", index=False)
    print(f"  -> {len(demand):,} rows "
          f"({demand['product_id'].nunique()} SKUs x {demand['center_id'].nunique()} centers x "
          f"{demand['date'].nunique()} days)")

    print("Generating production batch / expiry records...")
    batches = generate_production_batches(demand)
    batches.to_csv(RAW_DIR / "production_batches.csv", index=False)
    print(f"  -> {len(batches):,} batches "
          f"({(~batches['qc_pass']).sum()} failed QC and were not released)")

    print("Simulating inventory ledger ((s,S) periodic-review policy)...")
    inventory = simulate_inventory(demand, centers)
    inventory.to_parquet(RAW_DIR / "inventory_ledger.parquet", index=False)
    inventory.head(5000).to_csv(RAW_DIR / "inventory_ledger_sample.csv", index=False)
    total_demand = inventory["units_demanded"].sum()
    total_stockout = inventory["stockout_units"].sum()
    fill_rate = 1 - total_stockout / total_demand
    print(f"  -> {len(inventory):,} rows; network fill rate: {fill_rate:.2%}")

    print("Building raw-material supply network (materials, suppliers, BOM)...")
    supply_network = build_supply_network(products)
    supply_network["materials"].to_csv(RAW_DIR / "raw_materials.csv", index=False)
    supply_network["suppliers"].to_csv(RAW_DIR / "suppliers.csv", index=False)
    supply_network["bom"].to_csv(RAW_DIR / "bill_of_materials.csv", index=False)
    print(f"  -> {len(supply_network['materials'])} materials, "
          f"{len(supply_network['suppliers'])} suppliers, "
          f"{len(supply_network['bom'])} BOM lines")

    print("Simulating raw-material procurement (purchase orders + inventory)...")
    consumption = compute_material_consumption(batches, supply_network["bom"])
    purchase_orders, rm_inventory = simulate_procurement(
        consumption, supply_network["materials"], STUDY_START, STUDY_END
    )
    purchase_orders.to_csv(RAW_DIR / "purchase_orders.csv", index=False)
    rm_inventory.to_parquet(RAW_DIR / "raw_material_inventory.parquet", index=False)
    rm_inventory.head(5000).to_csv(RAW_DIR / "raw_material_inventory_sample.csv", index=False)
    print(f"  -> {len(purchase_orders):,} purchase orders "
          f"(network on-time delivery rate: {purchase_orders['on_time'].mean():.1%})")

    print(f"\nAll raw files written to: {RAW_DIR}")
    print(f"Total generation time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
