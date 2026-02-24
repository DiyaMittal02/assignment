"""
Lead Time Analysis Module
==========================
Assesses supply chain efficiency by analyzing procurement lead times:
    - Distribution analysis (overall and per supplier)
    - Trend over time (improving or worsening?)
    - Impact of lead time on procurement cost
    - Identification of bottleneck suppliers

Lead Time = ReceivingDate - PODate (days).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lead_time_analysis(purchases, output_dir=None):
    """
    Comprehensive lead time analysis.

    Parameters
    ----------
    purchases : pd.DataFrame
        Preprocessed purchases data with LeadTime column.
    output_dir : str, optional
        Directory to save visualizations.

    Returns
    -------
    dict
        Contains 'overall_stats', 'vendor_stats', and 'monthly_trend' DataFrames.
    """
    lt = purchases["LeadTime"].dropna()

    # ---- Overall statistics ----
    overall_stats = {
        "Mean": lt.mean(),
        "Median": lt.median(),
        "Std Dev": lt.std(),
        "Min": lt.min(),
        "Max": lt.max(),
        "P25": lt.quantile(0.25),
        "P75": lt.quantile(0.75),
        "P95": lt.quantile(0.95),
        "IQR": lt.quantile(0.75) - lt.quantile(0.25),
    }

    print("=" * 60)
    print("  LEAD TIME ANALYSIS")
    print("=" * 60)
    print(f"  Mean Lead Time           : {overall_stats['Mean']:.1f} days")
    print(f"  Median Lead Time         : {overall_stats['Median']:.1f} days")
    print(f"  Std Deviation            : {overall_stats['Std Dev']:.1f} days")
    print(f"  IQR (P25-P75)            : {overall_stats['P25']:.0f} - {overall_stats['P75']:.0f} days")
    print(f"  95th Percentile          : {overall_stats['P95']:.0f} days")
    print("-" * 60)

    # ---- Vendor-level lead time ----
    vendor_lt = (
        purchases.groupby("VendorName")["LeadTime"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    vendor_lt.columns = ["VendorName", "AvgLeadTime", "MedianLeadTime", "StdLeadTime", "OrderCount"]
    vendor_lt = vendor_lt[vendor_lt["OrderCount"] >= 5]  # filter low-volume vendors
    vendor_lt["Reliability"] = np.where(
        vendor_lt["StdLeadTime"] < vendor_lt["AvgLeadTime"] * 0.3, "Consistent",
        np.where(vendor_lt["StdLeadTime"] < vendor_lt["AvgLeadTime"] * 0.6, "Moderate", "Inconsistent")
    )
    vendor_lt = vendor_lt.sort_values("AvgLeadTime")

    # Flag bottleneck suppliers (top 10% by lead time)
    threshold = vendor_lt["AvgLeadTime"].quantile(0.90)
    bottlenecks = vendor_lt[vendor_lt["AvgLeadTime"] >= threshold]
    print(f"  Bottleneck Suppliers (>P90 lead time, >{threshold:.0f} days):")
    for _, row in bottlenecks.iterrows():
        print(f"    - {row['VendorName']}: {row['AvgLeadTime']:.1f} days avg "
              f"(+/- {row['StdLeadTime']:.1f})")

    # ---- Monthly trend ----
    monthly_lt = (
        purchases.groupby("YearMonth")["LeadTime"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    monthly_lt.columns = ["YearMonth", "AvgLeadTime", "MedianLeadTime", "StdLeadTime"]

    # Payment cycle analysis
    if "PayDate" in purchases.columns and "InvoiceDate" in purchases.columns:
        purchases_valid = purchases.dropna(subset=["PayDate", "InvoiceDate"])
        if len(purchases_valid) > 0:
            pay_cycle = (purchases_valid["PayDate"] - purchases_valid["InvoiceDate"]).dt.days
            pay_cycle = pay_cycle[pay_cycle >= 0]
            print(f"\n  Payment Cycle Analysis:")
            print(f"    Avg Invoice-to-Pay     : {pay_cycle.mean():.1f} days")
            print(f"    Median Invoice-to-Pay  : {pay_cycle.median():.1f} days")

    print("=" * 60)

    results = {
        "overall_stats": overall_stats,
        "vendor_stats": vendor_lt,
        "monthly_trend": monthly_lt,
    }

    if output_dir:
        _plot_lead_time(purchases, vendor_lt, monthly_lt, output_dir)

    return results


def _plot_lead_time(purchases, vendor_lt, monthly_lt, output_dir):
    """Generate lead time visualizations."""
    import os

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Lead Time Distribution (histogram)
    ax = axes[0, 0]
    lt_data = purchases["LeadTime"].dropna()
    ax.hist(lt_data, bins=50, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(lt_data.mean(), color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean ({lt_data.mean():.1f} days)")
    ax.axvline(lt_data.median(), color="#f39c12", linestyle="--", linewidth=2,
               label=f"Median ({lt_data.median():.1f} days)")
    ax.set_xlabel("Lead Time (days)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Lead Time Distribution", fontsize=13, fontweight="bold")
    ax.legend()

    # Chart 2: Monthly trend
    ax = axes[0, 1]
    months_str = monthly_lt["YearMonth"].astype(str)
    ax.plot(range(len(months_str)), monthly_lt["AvgLeadTime"],
            color="#2c3e50", linewidth=2, marker="o", markersize=5, label="Mean")
    ax.fill_between(
        range(len(months_str)),
        monthly_lt["AvgLeadTime"] - monthly_lt["StdLeadTime"],
        monthly_lt["AvgLeadTime"] + monthly_lt["StdLeadTime"],
        alpha=0.2, color="#3498db"
    )
    ax.plot(range(len(months_str)), monthly_lt["MedianLeadTime"],
            color="#e67e22", linewidth=1.5, linestyle="--", label="Median")
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=45, fontsize=7)
    ax.set_ylabel("Lead Time (days)", fontsize=11)
    ax.set_title("Lead Time Trend Over Time", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Chart 3: Top 15 vendors by lead time
    ax = axes[1, 0]
    top_vendors = vendor_lt.tail(15)
    colors_map = {"Consistent": "#27ae60", "Moderate": "#f39c12", "Inconsistent": "#e74c3c"}
    bar_colors = [colors_map.get(r, "#bdc3c7") for r in top_vendors["Reliability"]]
    ax.barh(range(len(top_vendors)), top_vendors["AvgLeadTime"],
            xerr=top_vendors["StdLeadTime"], color=bar_colors, alpha=0.85, capsize=3)
    ax.set_yticks(range(len(top_vendors)))
    ax.set_yticklabels(top_vendors["VendorName"], fontsize=8)
    ax.set_xlabel("Average Lead Time (days)", fontsize=10)
    ax.set_title("Slowest 15 Suppliers (color=reliability)", fontsize=12, fontweight="bold")

    # Chart 4: Box plot by vendor (top 10 by volume)
    ax = axes[1, 1]
    top_volume = vendor_lt.nlargest(10, "OrderCount")["VendorName"].tolist()
    box_data = [
        purchases[purchases["VendorName"] == v]["LeadTime"].dropna().values
        for v in top_volume
    ]
    bp = ax.boxplot(box_data, labels=[v[:20] for v in top_volume], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.7)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.set_ylabel("Lead Time (days)", fontsize=10)
    ax.set_title("Lead Time Distribution - Top 10 Vendors by Volume", fontsize=12, fontweight="bold")

    plt.suptitle("Lead Time Analysis Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lead_time_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[LeadTime] Charts saved to {output_dir}")
