"""
Network topology for Aurora Pharmaceuticals: 3 manufacturing plants
feeding 14 regional distribution centers (DCs). Center names and
regions are fictional/illustrative - this is a synthetic demo network,
not a real company's footprint.
"""
from __future__ import annotations

import pandas as pd

PLANTS = [
    {"plant_id": "PLANT-A", "plant_name": "Aurora Plant Alpha", "region": "North", "capacity_units_per_day": 55_000},
    {"plant_id": "PLANT-B", "plant_name": "Aurora Plant Beta", "region": "South", "capacity_units_per_day": 45_000},
    {"plant_id": "PLANT-C", "plant_name": "Aurora Plant Gamma", "region": "East", "capacity_units_per_day": 35_000},
]

# (center_id, center_name, region, market_size_weight, lead_time_days_from_plant)
_CENTERS = [
    ("DC-01", "Northgate Distribution Center", "North", 1.35, 3),
    ("DC-02", "Lakeside Distribution Center", "North", 1.05, 4),
    ("DC-03", "Summit Distribution Center", "North", 0.85, 5),
    ("DC-04", "Fairview Distribution Center", "North", 0.70, 4),
    ("DC-05", "Brookhaven Distribution Center", "South", 1.20, 3),
    ("DC-06", "Cedar Falls Distribution Center", "South", 0.95, 4),
    ("DC-07", "Ashford Distribution Center", "South", 0.75, 5),
    ("DC-08", "Kingsport Distribution Center", "South", 0.65, 6),
    ("DC-09", "Rosedale Distribution Center", "East", 1.10, 3),
    ("DC-10", "Elmwood Distribution Center", "East", 0.90, 4),
    ("DC-11", "Harborview Distribution Center", "East", 0.80, 4),
    ("DC-12", "Silverton Distribution Center", "East", 0.60, 6),
    ("DC-13", "Westfield Distribution Center", "East", 0.55, 6),
    ("DC-14", "Meridian Distribution Center", "South", 1.00, 4),
]

_CENTER_COLUMNS = [
    "center_id", "center_name", "region", "market_size_weight", "lead_time_days_from_plant",
]

# Which plant primarily supplies which region (single-sourcing per region
# keeps the batch/expiry simulation legible; real networks often multi-source).
REGION_TO_PLANT = {"North": "PLANT-A", "South": "PLANT-B", "East": "PLANT-C"}


def plants_dataframe() -> pd.DataFrame:
    return pd.DataFrame(PLANTS)


def centers_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(_CENTERS, columns=_CENTER_COLUMNS)
    df["supplying_plant_id"] = df["region"].map(REGION_TO_PLANT)
    return df
