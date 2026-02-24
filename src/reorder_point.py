"""
Reorder Point Analysis Module
==============================
Determines the inventory level at which a new order should be placed for
each product (brand) to avoid stockouts.

    ROP = (Average Daily Demand x Average Lead Time) + Safety Stock

Safety Stock uses the service-level approach:
    Safety Stock = Z * sigma_demand * sqrt(Lead Time)

Where Z = 1.65 for ~95% service level.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERVICE_LEVEL_Z = 1.65  # Z-score for 95% service level


def reorder_point(sales, purchases, output_dir=None):
    """
    Calculate the Reorder Point for each brand factoring in demand variability
    and lead time.

    Parameters
    ----------
    sales : pd.DataFrame
        Preprocessed sales data.
    purchases : pd.DataFrame
        Preprocessed purchases data with LeadTime column.
    output_dir : str, optional
        Directory to save visualizations.

    Returns
    -------
    pd.DataFrame
        Reorder point analysis with safety stock and ROP for each brand.
    """
    # Daily demand statistics per brand
    daily_demand = (
        sales.groupby(["Brand", "SalesDate"])["SalesQuantity"]
        .sum()
        .reset_index()
    )

    demand_stats = (
        daily_demand.groupby("Brand")["SalesQuantity"]
        .agg(["mean", "std", "sum"])
        .reset_index()
    )
    demand_stats.columns = ["Brand", "AvgDailyDemand", "StdDailyDemand", "TotalDemand"]
    demand_stats["StdDailyDemand"].fillna(0, inplace=True)

    # Lead time statistics per brand (from purchases)
    if "Brand" in purchases.columns:
        lead_time_stats = (
            purchases.groupby("Brand")["LeadTime"]
            .agg(["mean", "std"])
            .reset_index()
        )
        lead_time_stats.columns = ["Brand", "AvgLeadTime", "StdLeadTime"]
    else:
        # Fallback: use global average
        avg_lt = purchases["LeadTime"].mean()
        lead_time_stats = demand_stats[["Brand"]].copy()
        lead_time_stats["AvgLeadTime"] = avg_lt
        lead_time_stats["StdLeadTime"] = purchases["LeadTime"].std()

    rop_df = demand_stats.merge(lead_time_stats, on="Brand", how="left")

    # Fill NaN lead times with global average
    global_avg_lt = purchases["LeadTime"].mean()
    global_std_lt = purchases["LeadTime"].std()
    rop_df["AvgLeadTime"].fillna(global_avg_lt, inplace=True)
    rop_df["StdLeadTime"].fillna(global_std_lt, inplace=True)

    # Safety Stock = Z * sqrt(LT * sigma_d^2 + d_avg^2 * sigma_LT^2)
    rop_df["SafetyStock"] = SERVICE_LEVEL_Z * np.sqrt(
        rop_df["AvgLeadTime"] * rop_df["StdDailyDemand"]**2
        + rop_df["AvgDailyDemand"]**2 * rop_df["StdLeadTime"]**2
    )

    # Reorder Point
    rop_df["ReorderPoint"] = (
        rop_df["AvgDailyDemand"] * rop_df["AvgLeadTime"]
    ) + rop_df["SafetyStock"]

    rop_df["ReorderPoint"] = rop_df["ReorderPoint"].round(0).astype(int)
    rop_df["SafetyStock"] = rop_df["SafetyStock"].round(0).astype(int)

    rop_df = rop_df.sort_values("TotalDemand", ascending=False).reset_index(drop=True)

    # Summary
    print("=" * 60)
    print("  REORDER POINT ANALYSIS")
    print("=" * 60)
    print(f"  Service Level            : 95% (Z = {SERVICE_LEVEL_Z})")
    print(f"  Global Avg Lead Time     : {global_avg_lt:.1f} days")
    print(f"  Brands Analyzed          : {len(rop_df)}")
    print(f"  Avg Reorder Point        : {rop_df['ReorderPoint'].mean():.0f} units")
    print(f"  Avg Safety Stock         : {rop_df['SafetyStock'].mean():.0f} units")
    print("=" * 60)

    if output_dir:
        _plot_rop(rop_df, output_dir)

    return rop_df


def _plot_rop(rop_df, output_dir):
    """Generate Reorder Point visualizations."""
    import os

    top = rop_df.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Chart 1: ROP components stacked bar
    ax = axes[0]
    demand_component = top["AvgDailyDemand"] * top["AvgLeadTime"]
    x = np.arange(len(top))
    ax.barh(x, demand_component, label="Demand during LT", color="#3498db", alpha=0.85)
    ax.barh(x, top["SafetyStock"], left=demand_component, label="Safety Stock", color="#e74c3c", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(top["Brand"], fontsize=8)
    ax.set_xlabel("Units", fontsize=11)
    ax.set_title("Reorder Point Components - Top 20", fontsize=13, fontweight="bold")
    ax.legend()
    ax.invert_yaxis()

    # Chart 2: Safety stock vs demand variability
    ax = axes[1]
    scatter = ax.scatter(
        rop_df["StdDailyDemand"], rop_df["SafetyStock"],
        c=rop_df["AvgLeadTime"], cmap="coolwarm",
        alpha=0.6, edgecolors="white", linewidth=0.5, s=50
    )
    ax.set_xlabel("Demand Std Dev (daily)", fontsize=11)
    ax.set_ylabel("Safety Stock (units)", fontsize=11)
    ax.set_title("Safety Stock vs Demand Variability", fontsize=13, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Avg Lead Time (days)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reorder_point_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[ROP] Charts saved to {output_dir}")