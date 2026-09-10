"""
Upstream supply chain layer: the raw materials (APIs, excipients, and
packaging components) that feed production, the suppliers that
provide them, and the bill-of-materials (BOM) linking each finished
SKU to what it's made from.

Modeling choices (documented simplifications):
  * One dedicated API (active pharmaceutical ingredient) material per
    SKU - realistic, since each drug's active ingredient is unique.
  * A small shared pool of excipients (tableting fillers/binders) and
    packaging materials, since those genuinely are shared across many
    SKUs in real pharma manufacturing.
  * Quantities are abstracted as "material units consumed per 1,000
    finished units produced" rather than true per-tablet mg dosing -
    sufficient for supply-risk and procurement-timing analysis
    without requiring a full pharmaceutical formulation model.
  * Each material has exactly one primary supplier (single-sourcing),
    which is what makes API lead time such a realistic operational
    risk in this dataset - a delayed API shipment directly threatens
    a production batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPLIERS = [
    {"supplier_id": "SUP-01", "supplier_name": "Meridian API Imports", "region": "Overseas", "category": "API", "avg_lead_time_days": 55, "on_time_rate": 0.80},
    {"supplier_id": "SUP-02", "supplier_name": "Continental Pharma Ingredients", "region": "Overseas", "category": "API", "avg_lead_time_days": 48, "on_time_rate": 0.83},
    {"supplier_id": "SUP-03", "supplier_name": "Highland Fine Chemicals", "region": "Regional", "category": "API", "avg_lead_time_days": 30, "on_time_rate": 0.88},
    {"supplier_id": "SUP-04", "supplier_name": "Vantage Biosynthesis", "region": "Overseas", "category": "API", "avg_lead_time_days": 60, "on_time_rate": 0.76},
    {"supplier_id": "SUP-05", "supplier_name": "Northgate Excipients Co.", "region": "Domestic", "category": "Excipient", "avg_lead_time_days": 10, "on_time_rate": 0.95},
    {"supplier_id": "SUP-06", "supplier_name": "Cedar Packaging Supplies", "region": "Domestic", "category": "Packaging", "avg_lead_time_days": 8, "on_time_rate": 0.96},
    {"supplier_id": "SUP-07", "supplier_name": "Rosedale Glass & Vials", "region": "Domestic", "category": "Packaging", "avg_lead_time_days": 14, "on_time_rate": 0.92},
    {"supplier_id": "SUP-08", "supplier_name": "Fairview Specialty Chemicals", "region": "Regional", "category": "Excipient", "avg_lead_time_days": 18, "on_time_rate": 0.90},
]

_EXCIPIENTS = [
    ("MAT-EXC-01", "Lactose Monohydrate", "SUP-05"),
    ("MAT-EXC-02", "Microcrystalline Cellulose", "SUP-05"),
    ("MAT-EXC-03", "Magnesium Stearate", "SUP-08"),
    ("MAT-EXC-04", "Maize Starch", "SUP-08"),
    ("MAT-EXC-05", "Talc Powder", "SUP-05"),
]

_PACKAGING = [
    ("MAT-PKG-01", "PVC-Alu Blister Foil", "SUP-06"),
    ("MAT-PKG-02", "HDPE Bottles (100ct)", "SUP-06"),
    ("MAT-PKG-03", "Glass Vials (2mL)", "SUP-07"),
    ("MAT-PKG-04", "Cartons & Patient Leaflets", "SUP-06"),
]

_API_SUPPLIER_CYCLE = ["SUP-01", "SUP-02", "SUP-03", "SUP-04"]


def _dosage_form(row) -> str:
    name = row["generic_name"].lower()
    if "injection" in name or "insulin" in name:
        return "vial"
    if "syrup" in name:
        return "liquid"
    if "inhaler" in name:
        return "inhaler"
    if "cream" in name:
        return "tube"
    return "tablet_capsule"


_PACKAGING_BY_FORM = {
    "tablet_capsule": "MAT-PKG-01",  # blister
    "liquid": "MAT-PKG-02",          # bottle
    "vial": "MAT-PKG-03",            # glass vial
    "inhaler": "MAT-PKG-04",         # carton (device itself out of scope)
    "tube": "MAT-PKG-04",
}


def build_supply_network(products: pd.DataFrame, seed: int = 33) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    suppliers = pd.DataFrame(SUPPLIERS)

    api_rows = []
    for i, prod in enumerate(products.itertuples(index=False)):
        supplier = _API_SUPPLIER_CYCLE[i % len(_API_SUPPLIER_CYCLE)]
        api_rows.append({
            "material_id": f"MAT-API-{i+1:03d}",
            "material_name": f"{prod.generic_name.split()[0]} API",
            "material_category": "API",
            "supplier_id": supplier,
            "unit_cost": round(prod.unit_cost * rng.uniform(2.5, 4.0), 3),  # API is the priciest input
        })
    apis = pd.DataFrame(api_rows)

    excipients = pd.DataFrame(_EXCIPIENTS, columns=["material_id", "material_name", "supplier_id"])
    excipients["material_category"] = "Excipient"
    excipients["unit_cost"] = rng.uniform(0.005, 0.02, size=len(excipients)).round(4)

    packaging = pd.DataFrame(_PACKAGING, columns=["material_id", "material_name", "supplier_id"])
    packaging["material_category"] = "Packaging"
    packaging["unit_cost"] = rng.uniform(0.01, 0.08, size=len(packaging)).round(4)

    materials = pd.concat([apis, excipients, packaging], ignore_index=True)
    materials = materials.merge(
        suppliers[["supplier_id", "avg_lead_time_days", "on_time_rate", "region"]], on="supplier_id"
    )

    products_with_form = products.copy()
    products_with_form["dosage_form"] = products_with_form.apply(_dosage_form, axis=1)

    bom_rows = []
    for i, prod in enumerate(products_with_form.itertuples(index=False)):
        api_material_id = f"MAT-API-{i+1:03d}"
        bom_rows.append({"product_id": prod.product_id, "material_id": api_material_id,
                          "qty_per_1000_units": round(rng.uniform(8, 25), 2)})

        if prod.dosage_form == "tablet_capsule":
            excip = rng.choice(excipients["material_id"], size=2, replace=False)
            for m in excip:
                bom_rows.append({"product_id": prod.product_id, "material_id": m,
                                  "qty_per_1000_units": round(rng.uniform(150, 400), 1)})

        pkg_material_id = _PACKAGING_BY_FORM[prod.dosage_form]
        bom_rows.append({"product_id": prod.product_id, "material_id": pkg_material_id,
                          "qty_per_1000_units": round(rng.uniform(1000, 1050), 0)})

    bom = pd.DataFrame(bom_rows)

    return {"suppliers": suppliers, "materials": materials, "bom": bom}
