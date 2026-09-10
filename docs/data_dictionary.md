# Data Dictionary

## Provenance summary

**Everything in this repository is synthetically generated** for a
data-engineering / data-science portfolio project. "Aurora
Pharmaceuticals", its plants, distribution centers, and all sales,
inventory, and financial figures are fictional. Drug names are real
generic (INN) substance names, used only so that seasonal demand
patterns (e.g. antibiotics peaking in winter, allergy medication
peaking in spring/summer) are grounded in genuine epidemiological
behavior rather than arbitrary curves. No real company's sales,
pricing, manufacturing, or inventory data is used or implied.

## `products.csv` (42 rows)

| Column | Description |
|---|---|
| `product_id` | `SKU-001`...`SKU-042` |
| `generic_name` | Real INN generic drug name + strength |
| `category` | Therapeutic category (Analgesics, Antibiotics, Cardiovascular, Diabetes, Respiratory, Gastrointestinal, Vitamins, Endocrine, Anti-inflammatory) |
| `seasonality_profile` | Which seasonal curve the demand generator applies (see `docs/methodology.md`) |
| `base_daily_units_per_center` | Baseline daily demand rate used to seed the generator |
| `unit_cost` / `unit_price` | Synthetic unit economics (USD) |
| `pack_size` | Units per sales pack |
| `shelf_life_days` | Used to compute batch expiry dates |
| `requires_cold_chain` | True for insulin products |
| `margin_pct` | Derived: `(price - cost) / price` |

## `manufacturing_plants.csv` (3 rows) / `distribution_centers.csv` (14 rows)

Network topology: each of the 3 plants supplies one region (North/South/East);
each of the 14 distribution centers belongs to a region and has its own
`market_size_weight` (relative demand pull) and `lead_time_days_from_plant`.

## `daily_demand.parquet` (~430,000 rows)

One row per (product, distribution center, day) for 2023-01-01 -> 2024-12-31.

| Column | Description |
|---|---|
| `date` | ISO date |
| `product_id` / `center_id` | Foreign keys |
| `units_ordered` | Simulated daily order quantity (Negative-Binomial-distributed around a seasonal/trend/promo-adjusted mean) |

A 5,000-row sample (`daily_demand_sample.csv`) is included for quick
inspection without a Parquet reader.

## `production_batches.csv` (~1,750 rows)

| Column | Description |
|---|---|
| `batch_id` | Unique batch identifier |
| `product_id` / `plant_id` | Foreign keys |
| `production_date` / `expiry_date` | `expiry_date = production_date + shelf_life_days` |
| `quantity_produced` | Units in the batch |
| `qc_pass` | ~2% of batches fail QC and are never released to distribution |

## `raw_materials.csv` (51 rows) / `suppliers.csv` (8 rows) / `bill_of_materials.csv` (156 rows)

The upstream supply chain: each SKU has one dedicated API (active
ingredient) material, plus shared excipients (tableting fillers) and
packaging materials, each sourced from exactly one primary supplier.

| Column (`raw_materials.csv`) | Description |
|---|---|
| `material_id` | `MAT-API-###`, `MAT-EXC-##`, or `MAT-PKG-##` |
| `material_category` | API / Excipient / Packaging |
| `supplier_id` | Foreign key to `suppliers.csv` |
| `unit_cost` | Synthetic unit cost |
| `avg_lead_time_days` / `on_time_rate` | Denormalized from the supplier for convenience |

| Column (`suppliers.csv`) | Description |
|---|---|
| `region` | Domestic / Regional / Overseas - overseas API suppliers have the longest lead times (45-60 days) and lowest on-time rates (76-83%), which is the realistic root cause of production supply risk in this dataset |
| `avg_lead_time_days` / `on_time_rate` | Supplier reliability parameters used by the procurement simulation |

`bill_of_materials.csv` links `product_id` to the `material_id`s it consumes, in `qty_per_1000_units` (an abstracted consumption ratio rather than true per-tablet mg dosing - see `docs/methodology.md`).

## `purchase_orders.csv` (~730 rows) / `raw_material_inventory.parquet` (~37,000 rows)

Output of the raw-material procurement simulation
(`src/pharmapulse/data_generation/procurement.py`), driven by actual
production-batch consumption via the BOM.

| Column (`purchase_orders.csv`) | Description |
|---|---|
| `order_date` / `expected_delivery_date` / `actual_delivery_date` | Order timeline |
| `on_time` / `delay_days` | Whether the supplier's stated on-time rate held for this order |
| `quantity_ordered` / `unit_cost` | Order economics |

| Column (`raw_material_inventory.parquet`) | Description |
|---|---|
| `closing_stock` | End-of-day on-hand raw-material stock |
| `qty_consumed` | That day's consumption (from production batches) |

## `inventory_ledger.parquet` (~430,000 rows)

Output of the (s, S) periodic-review inventory simulation
(`src/pharmapulse/data_generation/inventory_sim.py`).

| Column | Description |
|---|---|
| `units_demanded` | Same as `units_ordered` above |
| `units_sold` | `units_demanded - stockout_units` |
| `closing_stock` | End-of-day on-hand inventory |
| `stockout_units` | Unmet demand that day |
| `units_received` / `units_reordered` | Replenishment arrivals / orders placed |

## Derived modeling dataset (`data/processed/weekly_features.parquet`)

Weekly-aggregated (product, center) series with calendar, lag, and
rolling-window features - see `src/pharmapulse/features/build_features.py`
for the exact feature list (`FEATURE_COLUMNS`).

## Working capital & supply-chain report outputs (`reports/`)

| File | Description |
|---|---|
| `working_capital.csv` | Daily DIO / DSO / DPO / CCC time series (60-day trailing window) |
| `supplier_performance.csv` | On-time delivery rate, average delay, and spend per supplier |
| `material_coverage.csv` | Days-of-cover vs. supplier lead time per raw material, with an `at_risk` flag |

## Report outputs (`reports/`)

| File | Description |
|---|---|
| `PharmaPulse_Planning_Report.xlsx` | 9-sheet planner workbook: Executive Summary (native charts), Forecast by Category, Inventory Plan, ABC-XYZ Segmentation, Expiry Risk, Working Capital (CCC), Suppliers, Raw Material Coverage, Model Performance |
| `pharmapulse_dashboard.html` | Standalone interactive dashboard (Plotly, red/orange theme), same underlying tables as the Excel report |
| `model_performance.csv`, `inventory_plan.csv`, `expiry_risk.csv` | Raw CSVs behind the report sheets |
