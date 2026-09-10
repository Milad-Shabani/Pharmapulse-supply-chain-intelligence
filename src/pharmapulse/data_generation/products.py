"""
Static product catalog: 40 SKUs across 8 therapeutic categories,
using real generic (INN) drug substance names for plausibility.

Each product carries the attributes a real demand-planning system
would key off of: unit economics, pack size, shelf life, cold-chain
requirement, and a `seasonality_profile` tag that the demand
generator uses to decide *how* that SKU's sales should move through
the year (e.g. respiratory/antibiotic SKUs spike in winter, allergy
SKUs spike in spring/summer, chronic-disease SKUs are flat).
"""
from __future__ import annotations

import pandas as pd

# (generic_name, category, seasonality_profile, base_daily_units_per_center,
#  unit_cost, unit_price, pack_size, shelf_life_days, cold_chain)
_PRODUCTS = [
    # Analgesics & Pain Relief - mild winter bump (colds/flu comorbid pain)
    ("Paracetamol 500mg", "Analgesics", "winter_bump", 220, 0.03, 0.08, 20, 730, False),
    ("Ibuprofen 400mg", "Analgesics", "winter_bump", 160, 0.04, 0.10, 20, 730, False),
    ("Diclofenac 50mg", "Analgesics", "flat", 90, 0.05, 0.13, 20, 730, False),
    ("Aspirin 100mg", "Analgesics", "flat", 130, 0.02, 0.06, 30, 900, False),
    ("Naproxen 250mg", "Analgesics", "flat", 70, 0.06, 0.15, 20, 730, False),

    # Antibiotics - strong winter/respiratory-season spike
    ("Amoxicillin 500mg", "Antibiotics", "strong_winter", 140, 0.08, 0.22, 21, 730, False),
    ("Azithromycin 250mg", "Antibiotics", "strong_winter", 95, 0.15, 0.45, 6, 730, False),
    ("Ceftriaxone 1g Injection", "Antibiotics", "strong_winter", 40, 0.90, 2.20, 1, 540, False),
    ("Ciprofloxacin 500mg", "Antibiotics", "mild_winter", 75, 0.10, 0.28, 10, 730, False),

    # Cardiovascular - chronic, very flat, slow upward trend
    ("Atorvastatin 20mg", "Cardiovascular", "flat_chronic", 180, 0.06, 0.18, 30, 900, False),
    ("Amlodipine 5mg", "Cardiovascular", "flat_chronic", 170, 0.04, 0.12, 30, 900, False),
    ("Losartan 50mg", "Cardiovascular", "flat_chronic", 150, 0.05, 0.15, 30, 900, False),
    ("Warfarin 5mg", "Cardiovascular", "flat_chronic", 60, 0.07, 0.20, 30, 730, False),
    ("Furosemide 40mg", "Cardiovascular", "flat_chronic", 85, 0.03, 0.09, 30, 900, False),

    # Diabetes - chronic, flat, mild upward trend, insulin needs cold chain
    ("Metformin 500mg", "Diabetes", "flat_chronic", 200, 0.03, 0.10, 30, 900, False),
    ("Gliclazide 80mg", "Diabetes", "flat_chronic", 90, 0.06, 0.17, 30, 900, False),
    ("Insulin Glargine 100IU/mL", "Diabetes", "flat_chronic", 55, 4.50, 9.80, 1, 540, True),
    ("Sitagliptin 100mg", "Diabetes", "flat_chronic", 65, 0.35, 0.85, 30, 900, False),

    # Respiratory / Allergy - spring/summer allergy spike + winter respiratory spike
    ("Salbutamol Inhaler 100mcg", "Respiratory", "strong_winter", 70, 1.20, 2.80, 1, 730, False),
    ("Cetirizine 10mg", "Respiratory", "spring_summer", 130, 0.02, 0.07, 20, 900, False),
    ("Loratadine 10mg", "Respiratory", "spring_summer", 110, 0.03, 0.09, 20, 900, False),
    ("Montelukast 10mg", "Respiratory", "mild_winter", 60, 0.20, 0.55, 30, 730, False),
    ("Dextromethorphan Syrup 100mL", "Respiratory", "strong_winter", 85, 0.60, 1.40, 1, 540, False),

    # Gastrointestinal - mostly flat, mild winter (dietary indulgence) bump
    ("Omeprazole 20mg", "Gastrointestinal", "flat_chronic", 140, 0.05, 0.15, 30, 900, False),
    ("Famotidine 40mg", "Gastrointestinal", "flat", 70, 0.04, 0.12, 30, 900, False),
    ("Domperidone 10mg", "Gastrointestinal", "flat", 55, 0.05, 0.14, 30, 730, False),
    ("Loperamide 2mg", "Gastrointestinal", "mild_winter", 40, 0.03, 0.09, 12, 900, False),
    ("Oral Rehydration Salts", "Gastrointestinal", "strong_summer", 100, 0.10, 0.25, 1, 730, False),

    # Vitamins & Supplements - New-Year resolution bump + winter (immunity) bump
    ("Vitamin D3 1000IU", "Vitamins", "new_year_bump", 150, 0.02, 0.08, 30, 900, False),
    ("Vitamin C 500mg", "Vitamins", "winter_bump", 170, 0.02, 0.07, 30, 900, False),
    ("Multivitamin Tablets", "Vitamins", "new_year_bump", 140, 0.04, 0.13, 30, 900, False),
    ("Iron + Folic Acid", "Vitamins", "flat", 90, 0.03, 0.10, 30, 900, False),
    ("Calcium + Vitamin D", "Vitamins", "flat", 100, 0.03, 0.11, 30, 900, False),

    # Endocrine - chronic, flat
    ("Levothyroxine 100mcg", "Endocrine", "flat_chronic", 95, 0.05, 0.16, 30, 730, False),
    ("Levothyroxine 50mcg", "Endocrine", "flat_chronic", 60, 0.05, 0.16, 30, 730, False),

    # Anti-inflammatory / Steroids - flat with mild winter (respiratory comorbidity) bump
    ("Prednisolone 5mg", "Anti-inflammatory", "mild_winter", 50, 0.03, 0.09, 30, 730, False),
    ("Dexamethasone 0.5mg", "Anti-inflammatory", "mild_winter", 35, 0.04, 0.12, 30, 730, False),
    ("Hydrocortisone Cream 1%", "Anti-inflammatory", "flat", 45, 0.30, 0.75, 1, 730, False),

    # Additional chronic + acute long tail to round out the 40-SKU portfolio
    ("Clopidogrel 75mg", "Cardiovascular", "flat_chronic", 55, 0.12, 0.35, 30, 900, False),
    ("Metoprolol 50mg", "Cardiovascular", "flat_chronic", 65, 0.05, 0.15, 30, 900, False),
    ("Doxycycline 100mg", "Antibiotics", "mild_winter", 50, 0.07, 0.20, 10, 730, False),
    ("Insulin Aspart 100IU/mL", "Diabetes", "flat_chronic", 30, 4.80, 10.20, 1, 540, True),
]

_COLUMNS = [
    "generic_name", "category", "seasonality_profile", "base_daily_units_per_center",
    "unit_cost", "unit_price", "pack_size", "shelf_life_days", "requires_cold_chain",
]


def products_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(_PRODUCTS, columns=_COLUMNS)
    df.insert(0, "product_id", [f"SKU-{i+1:03d}" for i in range(len(df))])
    df["margin_pct"] = ((df["unit_price"] - df["unit_cost"]) / df["unit_price"] * 100).round(1)
    return df


if __name__ == "__main__":
    df = products_dataframe()
    print(f"{len(df)} SKUs across {df['category'].nunique()} categories")
    print(df["category"].value_counts())
