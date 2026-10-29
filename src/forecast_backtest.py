"""Clean the simulated sales data, unconstrain demand, run the LightGBM backtest
and the stockout risk flag. , and write dashboard_data.json.
Run after the generator: python src/forecast_backtest.py
"""
import json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

D = Path("data")

# ---------------- Project 1: demand forecasting ----------------
s = pd.read_csv(D / "fact_sales_inventory.csv", parse_dates=["week_start"])
sku = pd.read_csv(D / "dim_sku.csv")
promo = pd.read_csv(D / "promo_calendar.csv", parse_dates=["week_start"])

s["dc_id"] = s.dc_id.str.upper().str.replace(" ", "_")             # standardise codes
s = s.drop_duplicates(["week_start", "sku_id", "dc_id"])           # batch reruns
s["returns"] = (-s.units_sold).clip(lower=0)
s["units_sold"] = s.units_sold.mask(s.units_sold < 0)               # returns are not demand
s = s.sort_values(["sku_id", "dc_id", "week_start"])
g = s.groupby(["sku_id", "dc_id"])
s["imputed"] = s.units_sold.isna()
s["units_sold"] = g.units_sold.transform(lambda x: x.interpolate(limit_direction="both"))
# unconstrain: stockout weeks under-report demand
clean_avg = g.units_sold.transform(lambda x: x.where(s.loc[x.index, "stockout_flag"] == 0).rolling(4, min_periods=1).mean().ffill())
s["demand"] = np.where(s.stockout_flag == 1, np.maximum(s.units_sold, clean_avg * 1.1), s.units_sold)

s = s.merge(sku, on="sku_id").merge(promo, on=["week_start", "sku_id"], how="left")
s["promo_flag"] = s.promo_type.notna().astype(int)
s["woy"] = s.week_start.dt.isocalendar().week.astype(int)
s["flu"] = ((s.woy >= 45) | (s.woy <= 8)).astype(int)
g = s.groupby(["sku_id", "dc_id"]).demand
for L in (2, 3, 4, 52):                                              # 2-week-ahead horizon -> lags >= 2
    s[f"lag{L}"] = g.shift(L)
s["roll4"] = g.transform(lambda x: x.shift(2).rolling(4).mean())
for c in ("category", "dc_id", "promo_type", "sku_id"):
    s[c + "_c"] = s[c].astype("category")
feats = ["lag2", "lag3", "lag4", "lag52", "roll4", "woy", "flu", "promo_flag", "category_c", "dc_id_c", "promo_type_c", "sku_id_c", "unit_price_gbp"]

cut = s.week_start.max() - pd.Timedelta(weeks=26)
tr, te = s[(s.week_start <= cut) & s.lag52.notna()], s[s.week_start > cut].copy()
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(tr[feats], tr.demand)
te["fc"] = m.predict(te[feats]).clip(0)
te["naive"] = te.lag52
wape = lambda a, f: float(np.abs(a - f).sum() / a.sum())
w_naive, w_lgb = wape(te.demand, te.naive), wape(te.demand, te.fc)
bias = float((te.fc.sum() - te.demand.sum()) / te.demand.sum())
# risk flag: forecast demand exceeds stock available two weeks earlier
te["stock_lag"] = s.groupby(["sku_id", "dc_id"]).closing_stock.shift(1).loc[te.index]
# flag when stock on hand covers less than 0.4 weeks of forecast demand
te["risk"] = (te.stock_lag < 0.4 * te.fc).astype(int)
tp = ((te.risk == 1) & (te.stockout_flag == 1)).sum()
recall = float(tp / max(1, te.stockout_flag.sum())); precision = float(tp / max(1, te.risk.sum()))

wk = te.groupby("week_start").agg(actual=("demand", "sum"), forecast=("fc", "sum"), naive=("naive", "sum")).reset_index()
hist = s[s.week_start > cut - pd.Timedelta(weeks=52)].groupby("week_start").demand.sum()
cat_w = te.groupby("category").apply(lambda d: wape(d.demand, d.fc)).sort_values()
latest = te[te.week_start == te.week_start.max()]
atrisk = (latest.assign(cover=lambda d: d.stock_lag / d.fc.replace(0, np.nan))
          .sort_values("cover").head(8)[["sku_id", "dc_id", "category", "stock_lag", "fc", "cover"]])
so_dc = s.groupby("dc_id").stockout_flag.mean().sort_values(ascending=False)
uplift = (s.groupby(["category", "promo_flag"]).demand.mean().unstack().pipe(lambda d: d[1] / d[0] - 1)).sort_values(ascending=False)

p1 = dict(
    wape_naive=w_naive, wape_lgb=w_lgb, bias=bias, recall=recall, precision=precision,
    stockout_rate=float(s.stockout_flag.mean()), rows=len(s), imputed=int(s.imputed.sum()),
    weekly=[dict(w=d.strftime("%Y-%m-%d"), a=round(a), f=round(f), n=round(n)) for d, a, f, n in wk.itertuples(index=False)],
    history=[dict(w=d.strftime("%Y-%m-%d"), a=round(v)) for d, v in hist.items() if d <= cut],
    cat_wape={k: round(v, 3) for k, v in cat_w.items()},
    stockout_by_dc={k: round(v, 3) for k, v in so_dc.items()},
    promo_uplift={k: round(v, 3) for k, v in uplift.items()},
    at_risk=[dict(sku=r.sku_id, dc=r.dc_id, cat=r.category, stock=round(r.stock_lag), fc=round(r.fc), cover=round(r.cover, 2)) for r in atrisk.itertuples()],
)

json.dump(p1, open("dashboard_data_forecast.json", "w"), indent=1)
print(json.dumps({k: v for k, v in p1.items() if not isinstance(v, (list, dict))}, indent=1))
