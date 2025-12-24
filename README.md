# Sales Demand Forecasting

This project analyzes retail sales data to forecast monthly demand and compare multiple forecasting approaches.

## Dataset
Retail sales dataset from Kaggle (not included in this repository).

## Methods
The following forecasting methods were evaluated:
- Naive Forecast
- Simple Moving Average (SMA)
- Simple Exponential Smoothing (SES)
- Holt’s Linear Trend

Models were compared using MAE and MAPE on a hold-out test set.

## Results
Simple Exponential Smoothing (SES) achieved the lowest forecast error and was selected as the final model.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib statsmodels
