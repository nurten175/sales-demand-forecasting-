#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Model Comparison (Monthly Demand)

Dataset: retail_sales_dataset.csv (Kaggle)
Columns used:
- Date
- Quantity

What this script does:
1) Sets working directory to Downloads (where CSV is)
2) Loads CSV
3) Aggregates monthly demand (Quantity)
4) Drops last month (often incomplete)
5) Train/Test split
6) Forecasts (Naive, SMA, SES, Holt)
7) Compares MAE/MAPE + plots

If SES/Holt errors with "No module named statsmodels":
    pip install statsmodels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) WHERE IS THE CSV?
# =========================
# You said the CSV is in Downloads:
os.chdir("/Users/nurtenerust/Downloads")

CSV_FILE = "retail_sales_dataset.csv"
DATE_COL = "Date"
TARGET_COL = "Quantity"


# =========================
# 1) LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

# Parse Date safely
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL, TARGET_COL])

print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())


# =========================
# 2) MONTHLY AGGREGATION
# =========================
monthly = (
    df.groupby(pd.Grouper(key=DATE_COL, freq="M"))[TARGET_COL]
      .sum()
      .sort_index()
)

# Very common issue: last month is incomplete -> drop it
if len(monthly) >= 2:
    monthly = monthly.iloc[:-1]

print("\nMonthly demand points:", len(monthly))
print(monthly.head())


# =========================
# 3) TRAIN / TEST SPLIT
# =========================
if len(monthly) < 6:
    raise ValueError(
        f"Not enough monthly points ({len(monthly)}) for train/test. Need ~6+."
    )

train_ratio = 0.8
train_size = int(len(monthly) * train_ratio)

train = monthly.iloc[:train_size].copy()
test = monthly.iloc[train_size:].copy()
h = len(test)  # horizon

print("\nTRAIN:", train.index.min(), "->", train.index.max(), "| n=", len(train))
print("TEST :", test.index.min(), "->", test.index.max(), "| n=", len(test))


# =========================
# 4) METRICS
# =========================
def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.mean(np.abs(y_true.values - y_pred.values)))

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    denom = np.where(y_true.values == 0, np.nan, y_true.values)
    return float(np.nanmean(np.abs((y_true.values - y_pred.values) / denom)) * 100)


# =========================
# 5) FORECAST METHODS
# =========================
def forecast_naive(train_series: pd.Series, steps: int) -> pd.Series:
    last_val = float(train_series.iloc[-1])
    return pd.Series([last_val] * steps)

def forecast_sma(train_series: pd.Series, steps: int, window: int = 3) -> pd.Series:
    # iterative multi-step SMA (uses last window values, appends predictions)
    history = train_series.astype(float).reset_index(drop=True)
    preds = []
    for _ in range(steps):
        pred = float(history.iloc[-window:].mean())
        preds.append(pred)
        history = pd.concat([history, pd.Series([pred])], ignore_index=True)
    return pd.Series(preds)

def forecast_ses(train_series: pd.Series, steps: int) -> pd.Series:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(train_series.astype(float), initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return pd.Series(fc.values)

def forecast_holt(train_series: pd.Series, steps: int) -> pd.Series:
    from statsmodels.tsa.holtwinters import Holt
    model = Holt(train_series.astype(float), initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return pd.Series(fc.values)


# =========================
# 6) RUN FORECASTS
# =========================
preds = {}

# Naive
preds["Naive"] = forecast_naive(train, h)

# SMA
SMA_WINDOW = 3
preds[f"SMA_{SMA_WINDOW}"] = forecast_sma(train, h, window=SMA_WINDOW)

# SES + Holt (may require statsmodels)
try:
    preds["SES"] = forecast_ses(train, h)
    preds["Holt"] = forecast_holt(train, h)
except Exception as e:
    print("\n[INFO] SES/Holt skipped:", repr(e))
    print("[INFO] If you want SES/Holt, install statsmodels: pip install statsmodels\n")

# Put test index onto forecasts
for k in list(preds.keys()):
    preds[k] = pd.Series(preds[k].values, index=test.index)


# =========================
# 7) SCORE MODELS
# =========================
rows = []
for name, yhat in preds.items():
    rows.append(
        {"model": name, "MAE": mae(test, yhat), "MAPE_%": mape(test, yhat)}
    )

results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

print("\n=== MODEL COMPARISON (lower MAE is better) ===")
print(results)

best_model = results.loc[0, "model"]
print("\nBEST MODEL:", best_model)


# =========================
# 8) PLOT RESULTS
# =========================
plt.figure(figsize=(11, 5))
plt.plot(train.index, train.values, label="Train (Actual)")
plt.plot(test.index, test.values, label="Test (Actual)")

for name, yhat in preds.items():
    plt.plot(yhat.index, yhat.values, linestyle="--", label=f"{name} Forecast")

plt.title("Monthly Demand: Actual vs Forecasts (Test Period)")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.legend()
plt.show()

# =========================
# 9) FINAL FORECAST (SES)
# =========================
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# SES modelini tüm veride yeniden eğit (final model)
ses_model = SimpleExpSmoothing(monthly.astype(float), initialization_method="estimated")
ses_fit = ses_model.fit(optimized=True)

# Kaç ay ileri tahmin?
FORECAST_MONTHS = 3
future_forecast = ses_fit.forecast(FORECAST_MONTHS)

print("\n=== FUTURE DEMAND FORECAST (SES) ===")
print(future_forecast)



