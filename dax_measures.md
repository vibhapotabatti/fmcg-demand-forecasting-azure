# DAX measures

## Project 1 · Demand forecasting (tables: fact_forecast, fact_inventory, dim_sku, dim_dc, dim_date)

```DAX
Actual Demand = SUM ( fact_forecast[demand] )
Forecast = SUM ( fact_forecast[forecast] )
Baseline = SUM ( fact_forecast[naive] )

Abs Error = SUMX ( fact_forecast, ABS ( fact_forecast[demand] - fact_forecast[forecast] ) )
WAPE = DIVIDE ( [Abs Error], [Actual Demand] )
WAPE Baseline = DIVIDE ( SUMX ( fact_forecast, ABS ( fact_forecast[demand] - fact_forecast[naive] ) ), [Actual Demand] )
Forecast Bias = DIVIDE ( [Forecast] - [Actual Demand], [Actual Demand] )

Closing Stock = SUM ( fact_inventory[closing_stock] )
Weeks of Cover = DIVIDE ( [Closing Stock], [Forecast] )
Risk Status =
    SWITCH ( TRUE (),
        [Weeks of Cover] < 0.4, "Stockout",
        [Weeks of Cover] < 1.5, "At risk",
        "OK" )

Stockout Rate = AVERAGE ( fact_inventory[stockout_flag] )
Demand Rolling 4W =
    CALCULATE ( [Actual Demand],
        DATESINPERIOD ( dim_date[date], MAX ( dim_date[date] ), -28, DAY ) )
Promo Uplift =
    VAR promo = CALCULATE ( AVERAGE ( fact_forecast[demand] ), fact_forecast[promo_flag] = 1 )
    VAR base  = CALCULATE ( AVERAGE ( fact_forecast[demand] ), fact_forecast[promo_flag] = 0 )
    RETURN DIVIDE ( promo, base ) - 1
```

## Build checklist in Power BI Desktop
1. Get Data: SQL Server (Project 2 view) or Parquet / Databricks (Project 1 gold tables).
2. Build a dim_date table (`CALENDARAUTO()`), mark it as the date table, relate it to every fact.
3. Model view: star schema, single-direction relationships from dims to facts.
4. Add the measures above in a dedicated `_Measures` table.
5. Pages: copy the layout of supply-chain-dashboard.html (KPI cards, line chart, table with conditional formatting, bar charts).
6. Add slicers (category, DC, customer, plant, date), a drill-through page per SKU, and tooltips.
7. Row-level security: a role filtering dim_dc[region] or dim_customer to show you know RLS.
8. Publish to the Power BI Service (free trial or a Fabric trial), set scheduled refresh, add a subscription or Power Automate alert.
