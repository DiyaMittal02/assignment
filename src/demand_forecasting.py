"""
Demand Forecasting Module
==========================
Uses SARIMAX time-series models to forecast future demand for the top-selling
brands. Automatically selects the best frequency (weekly preferred when
monthly data points are limited).

Approach:
    1. Identify top N brands by total sales volume
    2. Aggregate to weekly frequency (gives more data points)
    3. Fit SARIMAX(1,1,1) model per brand
    4. Forecast ahead with 95% confidence intervals
    5. Evaluate in-sample fit with MAE and RMSE
"""

import warnings
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def forecast_top_brands(sales, n_brands=10, forecast_steps=4, output_dir=None):
    """
    Forecast demand for the top N brands using SARIMAX.

    Uses weekly aggregation to maximise data points available from the
    ~2-month dataset. Falls back gracefully when data is too sparse.

    Parameters
    ----------
    sales : pd.DataFrame
        Preprocessed sales data with SalesDate and SalesQuantity columns.
    n_brands : int
        Number of top brands to forecast (default: 10).
    forecast_steps : int
        Periods to forecast ahead (default: 4 weeks).
    output_dir : str, optional
        Directory to save visualizations and results.

    Returns
    -------
    dict
        {brand: dict with 'history', 'forecast', 'mae', 'rmse', 'aic'}
    """
    results_dict = {}
    metrics_list = []

    top_brands = (
        sales.groupby("Brand")["SalesQuantity"]
        .sum()
        .sort_values(ascending=False)
        .index[:n_brands]
    )

    print("=" * 60)
    print("  DEMAND FORECASTING (SARIMAX - Weekly)")
    print("=" * 60)
    print(f"  Top {n_brands} brands selected for forecasting")
    print(f"  Forecast horizon: {forecast_steps} weeks")
    print("-" * 60)

    for brand in top_brands:
        df = sales[sales["Brand"] == brand]

        # Use weekly aggregation for better data density
        weekly = (
            df.groupby(pd.Grouper(key="SalesDate", freq="W"))["SalesQuantity"]
            .sum()
            .fillna(0)
        )

        # Need at least 4 non-zero weeks for modeling
        if len(weekly[weekly > 0]) < 4:
            print(f"  {brand:<30} Skipped (insufficient data: "
                  f"{len(weekly[weekly > 0])} weeks)")
            continue

        try:
            model = SARIMAX(
                weekly,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fit = model.fit(disp=False, maxiter=200)
            forecast_obj = fit.get_forecast(steps=forecast_steps)

            forecast_df = pd.DataFrame({
                "Predicted": forecast_obj.predicted_mean,
                "Lower_CI": forecast_obj.conf_int().iloc[:, 0],
                "Upper_CI": forecast_obj.conf_int().iloc[:, 1],
            })

            # Clamp negatives to 0
            forecast_df = forecast_df.clip(lower=0)

            # In-sample metrics
            fitted = fit.fittedvalues
            residuals = weekly - fitted
            mae = np.abs(residuals).mean()
            rmse = np.sqrt((residuals**2).mean())

            results_dict[brand] = {
                "history": weekly,
                "forecast": forecast_df,
                "mae": mae,
                "rmse": rmse,
                "aic": fit.aic,
            }

            metrics_list.append({
                "Brand": brand,
                "DataPoints": len(weekly),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "AIC": round(fit.aic, 2),
                "Forecast_AvgWeekly": round(forecast_df["Predicted"].mean(), 0),
            })

            print(f"  {brand:<30} MAE={mae:>8.1f}  RMSE={rmse:>8.1f}  "
                  f"AIC={fit.aic:>10.1f}  ({len(weekly)} weeks)")

        except Exception as e:
            print(f"  {brand:<30} Failed: {str(e)[:50]}")
            continue

    print(f"\n  Successfully forecasted: {len(results_dict)} / {n_brands} brands")
    print("=" * 60)

    if output_dir and results_dict:
        _plot_forecasts(results_dict, output_dir)
        # Save metrics table
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(
                os.path.join(output_dir, "forecast_metrics.csv"), index=False
            )
            print(f"[Forecast] Metrics saved to {output_dir}/forecast_metrics.csv")

    return results_dict


def _plot_forecasts(results_dict, output_dir):
    """Generate forecast visualizations - individual subplots."""
    brands = list(results_dict.keys())
    n = len(brands)

    if n == 0:
        return

    cols = min(2, n)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)

    for idx, brand in enumerate(brands):
        row_i, col_i = divmod(idx, cols)
        ax = axes[row_i][col_i]
        data = results_dict[brand]
        history = data["history"]
        forecast = data["forecast"]

        # Plot historical
        ax.plot(history.index, history.values, color="#2c3e50",
                linewidth=1.5, label="Historical")

        # Plot forecast
        fc_idx = forecast.index
        ax.plot(fc_idx, forecast["Predicted"], color="#e74c3c",
                linewidth=2, marker="o", markersize=5, label="Forecast")

        # Confidence interval
        ax.fill_between(
            fc_idx,
            forecast["Lower_CI"],
            forecast["Upper_CI"],
            alpha=0.2, color="#e74c3c", label="95% CI"
        )

        ax.set_title(f"{brand}\nMAE={data['mae']:.0f}, RMSE={data['rmse']:.0f}",
                      fontsize=10, fontweight="bold")
        ax.set_ylabel("Weekly Qty", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row_i, col_i = divmod(idx, cols)
        axes[row_i][col_i].set_visible(False)

    plt.suptitle("SARIMAX Demand Forecasts (Weekly) - Top Brands",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demand_forecasts.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Forecast] Charts saved to {output_dir}")