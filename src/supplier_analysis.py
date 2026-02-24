"""
Supplier Performance Analysis Module
=====================================
Evaluates vendor performance across multiple KPIs:
    - Total spend and quantity supplied
    - Average lead time and reliability
    - Cost efficiency (unit cost)
    - Delivery consistency (lead time standard deviation)

Ranks suppliers to identify strategic partners vs. underperformers.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def supplier_performance(purchases, output_dir=None):
    """
    Analyse supplier performance on cost, volume, and delivery metrics.

    Parameters
    ----------
    purchases : pd.DataFrame
        Preprocessed purchases data.
    output_dir : str, optional
        Directory to save visualizations.

    Returns
    -------
    pd.DataFrame
        Supplier scorecard sorted by total spend.
    """
    supplier_summary = (
        purchases.groupby("VendorName")
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Total_Spent=("Dollars", "sum"),
            Num_Orders=("Quantity", "count"),
            Avg_LeadTime=("LeadTime", "mean"),
            Std_LeadTime=("LeadTime", "std"),
            Min_LeadTime=("LeadTime", "min"),
            Max_LeadTime=("LeadTime", "max"),
        )
        .reset_index()
    )

    supplier_summary["Avg_Cost_Per_Unit"] = (
        supplier_summary["Total_Spent"] / supplier_summary["Total_Quantity"]
    )

    # Delivery reliability: coefficient of variation of lead time
    supplier_summary["LeadTime_CV"] = (
        supplier_summary["Std_LeadTime"] / supplier_summary["Avg_LeadTime"]
    ).fillna(0)

    # Scoring: lower lead time + lower CV = better
    # Normalize metrics for scoring
    max_lt = supplier_summary["Avg_LeadTime"].max()
    max_cv = supplier_summary["LeadTime_CV"].max()

    supplier_summary["DeliveryScore"] = (
        100 - (
            (supplier_summary["Avg_LeadTime"] / max_lt * 50) +
            (supplier_summary["LeadTime_CV"] / max_cv * 50) if max_cv > 0
            else supplier_summary["Avg_LeadTime"] / max_lt * 100
        )
    ).clip(0, 100).round(1)

    supplier_summary = supplier_summary.sort_values("Total_Spent", ascending=False).reset_index(drop=True)

    # Summary
    print("=" * 60)
    print("  SUPPLIER PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"  Total Suppliers          : {len(supplier_summary)}")
    print(f"  Total Procurement Spend  : ${supplier_summary['Total_Spent'].sum():,.2f}")
    print(f"  Total Units Procured     : {supplier_summary['Total_Quantity'].sum():,.0f}")
    print(f"  Avg Lead Time (all)      : {purchases['LeadTime'].mean():.1f} days")
    print(f"  Top Supplier by Spend    : {supplier_summary.iloc[0]['VendorName']}")
    print("=" * 60)

    if output_dir:
        _plot_suppliers(supplier_summary, output_dir)

    return supplier_summary


def _plot_suppliers(supplier_summary, output_dir):
    """Generate supplier analysis visualizations."""
    import os

    top = supplier_summary.head(15)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Top suppliers by spend
    ax = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))
    ax.barh(range(len(top)), top["Total_Spent"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["VendorName"], fontsize=8)
    ax.set_xlabel("Total Spend ($)", fontsize=10)
    ax.set_title("Top 15 Suppliers by Spend", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Chart 2: Lead Time distribution
    ax = axes[0, 1]
    ax.barh(range(len(top)), top["Avg_LeadTime"], xerr=top["Std_LeadTime"],
            color="#e67e22", alpha=0.8, capsize=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["VendorName"], fontsize=8)
    ax.set_xlabel("Average Lead Time (days)", fontsize=10)
    ax.set_title("Lead Time by Supplier (with Std Dev)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Chart 3: Cost per unit
    ax = axes[1, 0]
    ax.barh(range(len(top)), top["Avg_Cost_Per_Unit"], color="#27ae60", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["VendorName"], fontsize=8)
    ax.set_xlabel("Average Cost Per Unit ($)", fontsize=10)
    ax.set_title("Cost Efficiency by Supplier", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Chart 4: Volume vs Lead Time scatter
    ax = axes[1, 1]
    scatter = ax.scatter(
        supplier_summary["Avg_LeadTime"],
        supplier_summary["Total_Quantity"],
        s=supplier_summary["Total_Spent"] / supplier_summary["Total_Spent"].max() * 500,
        c=supplier_summary["Avg_Cost_Per_Unit"], cmap="RdYlGn_r",
        alpha=0.7, edgecolors="gray", linewidth=0.5
    )
    ax.set_xlabel("Avg Lead Time (days)", fontsize=10)
    ax.set_ylabel("Total Quantity Supplied", fontsize=10)
    ax.set_title("Supplier Landscape (size=spend, color=unit cost)", fontsize=12, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Avg Unit Cost ($)")

    plt.suptitle("Supplier Performance Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "supplier_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Supplier] Charts saved to {output_dir}")