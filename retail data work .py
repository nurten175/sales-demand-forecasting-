"""
Forecast Model Comparison (Monthly Demand)

Dataset: retail_sales_dataset.csv (Kaggle)
Used columns:
- Date
- Quantity

What this script does:
1) Loads CSV (not included in repo)
2) Aggregates monthly demand (Quantity)
3) Drops last month (often incomplete in real-world exports)
4) Train/Test split
5) Forecasts: Naive, SMA, SES, Holt
6) Compares models with MAE/MAPE and plots results

How to run:
    pip install pandas numpy matplotlib statsmodels
    python forecast_compare.py

Note:
- Download the dataset from Kaggle and place `retail_sales_dataset.csv`
  next to this script OR update CSV_FILE below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CONFIG
# =========================
CSV_FILE = Path("retail_sales_dataset.csv")  # put CSV next to this script, or set a relative path like Path("data/retail_sales_dataset.csv")
DATE_COL = "Date"
TARGET_COL = "Quantity"

AGG_FREQ = "M"              # monthly aggregation
DROP_LAST_INCOMPLETE = True # drop last month to avoid partial-month bias

TRAIN_RATIO = 0.8
SMA_WINDOW = 3


# =========================
# METRICS
# =========================
def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.mean(np.abs(y_true.values - y_pred.values)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    denom = np.where(y_true.values == 0, np.nan, y_true.values)
    return float(np.nanmean(np.abs((y_true.values - y_pred.values) / denom)) * 100)


# =========================
# FORECAST METHODS
# =========================
def forecast_naive(train: pd.Series, steps: int) -> pd.Series:
    """Forecast all future points as the last observed value."""
    last_val = float(train.iloc[-1])
    return pd.Series([last_val] * steps)


def forecast_sma(train: pd.Series, steps: int, window: int = 3) -> pd.Series:
    """
    Multi-step Simple Moving Average forecast (iterative):
    - take mean of last `window` points
    - append prediction and repeat
    """
    history = train.astype(float).reset_index(drop=True)
    preds = []
    for _ in range(steps):
        pred = float(history.iloc[-window:].mean())
        preds.append(pred)
        history = pd.concat([history, pd.Series([pred])], ignore_index=True)
    return pd.Series(preds)


def forecast_ses(train: pd.Series, steps: int) -> pd.Series:
    """Simple Exponential Smoothing (level only)."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    model = SimpleExpSmoothing(train.astype(float), initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return pd.Series(fc.values)


def forecast_holt(train: pd.Series, steps: int) -> pd.Series:
    """Holt’s linear trend method."""
    from statsmodels.tsa.holtwinters import Holt

    model = Holt(train.astype(float), initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return pd.Series(fc.values)


# =========================
# MAIN
# =========================
def main() -> None:
    # 1) Load
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_FILE}\n"
            "Download it from Kaggle and place it next to this script, "
            "or update CSV_FILE in the CONFIG section."
        )

    df = pd.read_csv(CSV_FILE)

    # 2) Clean + parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL])

    # 3) Monthly aggregation
    monthly = (
        df.groupby(pd.Grouper(key=DATE_COL, freq=AGG_FREQ))[TARGET_COL]
          .sum()
          .sort_index()
    )

    if DROP_LAST_INCOMPLETE and len(monthly) >= 2:
        monthly = monthly.iloc[:-1]

    if len(monthly) < 6:
        raise ValueError(
            f"Not enough monthly points ({len(monthly)}) for a meaningful train/test split. "
            "Try using more data or aggregate differently."
        )

    print("Monthly demand points:", len(monthly))
    print(monthly.head())

    # 4) Train/Test split
    train_size = int(len(monthly) * TRAIN_RATIO)
    train = monthly.iloc[:train_size].copy()
    test = monthly.iloc[train_size:].copy()
    h = len(test)

    print("\nTRAIN:", train.index.min(), "->", train.index.max(), "| n=", len(train))
    print("TEST :", test.index.min(), "->", test.index.max(), "| n=", len(test))

    # 5) Forecasts
    preds = {
        "Naive": forecast_naive(train, h),
        f"SMA_{SMA_WINDOW}": forecast_sma(train, h, window=SMA_WINDOW),
    }

    # SES/Holt require statsmodels
    try:
        preds["SES"] = forecast_ses(train, h)
        preds["Holt"] = forecast_holt(train, h)
    except Exception as e:
        print("\n[INFO] SES/Holt skipped:", repr(e))
        print("[INFO] Install statsmodels to enable them: pip install statsmodels\n")

    # Put test index onto predictions
    for k in list(preds.keys()):
        preds[k] = pd.Series(preds[k].values, index=test.index)

    # 6) Score
    rows = []
    for name, yhat in preds.items():
        rows.append({"model": name, "MAE": mae(test, yhat), "MAPE_%": mape(test, yhat)})

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    print("\n=== MODEL COMPARISON (lower MAE is better) ===")
    print(results)

    best_model = results.loc[0, "model"]
    print("\nBEST MODEL:", best_model)

    # 7) Plot
    plt.figure(figsize=(11, 5))
    plt.plot(train.index, train.values, label="Train (Actual)")
    plt.plot(test.index, test.values, label="Test (Actual)")

    for name, yhat in preds.items():
        plt.plot(yhat.index, yhat.values, linestyle="--", label=f"{name} Forecast")

    plt.title("Monthly Demand: Actual vs Forecasts (Test Period)")
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


