"""Synthetic FMCG demand + inventory data (Project 1).
Generates weekly SKU x distribution-centre sales with seasonality, promotions,
flu-season spikes for hygiene SKUs, stockout censoring and messy source quirks.
Usage: python gen_demand_data.py  -> writes CSVs to ./data/
"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path("data"); OUT.mkdir(exist_ok=True)

skus = pd.DataFrame({
    "sku_id": [f"SKU{1000+i}" for i in range(40)],
    "category": rng.choice(["Disinfectant", "Handwash", "Surface Cleaner", "Pain Relief", "Dishwash"], 40),
    "pack_size": rng.choice(["250ml", "500ml", "1L", "12ct", "24ct"], 40),
    "unit_price_gbp": rng.uniform(1.5, 9.0, 40).round(2),
})
dcs = ["DC_NORTH", "DC_MIDLANDS", "DC_SOUTH", "DC_SCOTLAND"]
weeks = pd.date_range("2023-01-02", "2025-12-29", freq="W-MON")

rows = []
for _, s in skus.iterrows():
    base = rng.uniform(300, 3000)
    flu_sensitive = s.category in ("Disinfectant", "Handwash", "Pain Relief")
    promo_weeks = rng.random(len(weeks)) < 0.12  # national promos hit every DC in the same week
    for dc in dcs:
        dc_mult = rng.uniform(0.6, 1.4)
        stock = base * 3
        for t, wk in enumerate(weeks):
            woy = wk.isocalendar().week
            season = 1 + 0.15 * np.sin(2 * np.pi * woy / 52)
            flu = 1 + (0.5 if flu_sensitive and (woy >= 45 or woy <= 8) else 0)
            promo = promo_weeks[t]
            demand = base * dc_mult * season * flu * (1.35 if promo else 1) * (1 + 0.002 * t)
            demand = max(0, rng.normal(demand, demand * 0.15))
            # replenishment planned on last year's pattern, so flu-season and promo spikes still cause stockouts
            stock = min(stock + base * dc_mult * season * (1 + 0.35 * (flu > 1)) * rng.uniform(1.0, 1.5), base * 4)
            sold = min(demand, stock)  # censored by stock -> true demand hidden
            stock -= sold
            rows.append([wk.date(), s.sku_id, dc, round(sold), int(promo), round(stock), int(sold < demand)])

sales = pd.DataFrame(rows, columns=["week_start", "sku_id", "dc_id", "units_sold",
                                    "promo_flag", "closing_stock", "stockout_flag"])

# Inject realistic data-quality issues to clean later
idx = sales.sample(frac=0.02, random_state=1).index
sales.loc[idx, "units_sold"] = np.nan                         # missing EDI loads
dup = sales.sample(frac=0.005, random_state=2)                # duplicate batch reruns
sales = pd.concat([sales, dup])
neg = sales.sample(frac=0.003, random_state=3).index
sales.loc[neg, "units_sold"] = -sales.loc[neg, "units_sold"]  # returns booked as negatives
sales["dc_id"] = sales["dc_id"].where(rng.random(len(sales)) > 0.01,
                                      sales["dc_id"].str.lower().str.replace("_", " "))  # inconsistent codes

promos = sales[sales.promo_flag == 1][["week_start", "sku_id"]].drop_duplicates()
promos["promo_type"] = rng.choice(["Multibuy", "TPR 20%", "Feature+Display"], len(promos))
promos["retailer"] = rng.choice(["Tesco", "Sainsbury's", "Asda", "Boots"], len(promos))

skus.to_csv(OUT / "dim_sku.csv", index=False)
sales.drop(columns="promo_flag").to_csv(OUT / "fact_sales_inventory.csv", index=False)
promos.to_csv(OUT / "promo_calendar.csv", index=False)
print(sales.shape, "rows written to", OUT.resolve())
