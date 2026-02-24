"""
ABC Analysis Module
===================
Classifies inventory items (by Brand) into A / B / C categories based on
revenue contribution using the Pareto principle.

    - A : Top ~70% of cumulative revenue  (vital few)
    - B : Next ~20% (70-90%)              (moderate)
    - C : Bottom ~10% (90-100%)           (trivial many)

This analysis helps prioritize inventory management efforts and capital
allocation towards the highest-value products.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def abc_analysis(sales, output_dir=None):
    """
    Perform ABC classification on brands by revenue contribution.

    Parameters
    ----------
    sales : pd.DataFrame
        Preprocessed sales data with 'Brand' and 'Revenue' columns.
    output_dir : str, optional
        Directory to save visualizations.

    Returns
    -------
    pd.DataFrame
        Brand-level ABC classification with revenue stats.
    """
    brand_revenue = (
        sales.groupby("Brand")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("SalesQuantity", "sum"),
            Transactions=("Revenue", "count"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )

    brand_revenue["Revenue %"] = (
        brand_revenue["Revenue"] / brand_revenue["Revenue"].sum() * 100
    )
    brand_revenue["Cumulative %"] = brand_revenue["Revenue %"].cumsum()

    def classify(x):
        if x <= 70:
            return "A"
        elif x <= 90:
            return "B"
        return "C"

    brand_revenue["Category"] = brand_revenue["Cumulative %"].apply(classify)

    # Summary statistics
    summary = brand_revenue.groupby("Category").agg(
        Brand_Count=("Brand", "count"),
        Total_Revenue=("Revenue", "sum"),
        Avg_Revenue_Per_Brand=("Revenue", "mean"),
    )

    print("=" * 60)
    print("  ABC ANALYSIS SUMMARY")
    print("=" * 60)
    for cat in ["A", "B", "C"]:
        if cat in summary.index:
            row = summary.loc[cat]
            pct = row["Total_Revenue"] / brand_revenue["Revenue"].sum() * 100
            print(f"  Category {cat}: {int(row['Brand_Count']):>4} brands | "
                  f"${row['Total_Revenue']:>14,.2f} revenue ({pct:.1f}%)")
    print("=" * 60)

    if output_dir:
        _plot_abc(brand_revenue, output_dir)

    return brand_revenue


def _plot_abc(brand_revenue, output_dir):
    """Generate ABC analysis visualizations."""
    import os

    colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}

    # --- Chart 1: Pareto Chart ---
    fig, ax1 = plt.subplots(figsize=(14, 6))

    top_n = brand_revenue.head(30)
    bar_colors = [colors[c] for c in top_n["Category"]]

    ax1.bar(range(len(top_n)), top_n["Revenue"], color=bar_colors, alpha=0.8)
    ax1.set_ylabel("Revenue ($)", fontsize=12)
    ax1.set_xlabel("Brand (ranked by revenue)", fontsize=12)
    ax1.set_title("ABC Pareto Analysis - Top 30 Brands", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(top_n)))
    ax1.set_xticklabels(top_n["Brand"], rotation=45, ha="right", fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(range(len(top_n)), top_n["Cumulative %"], color="#2c3e50",
             linewidth=2.5, marker="o", markersize=4)
    ax2.axhline(y=70, color="#2ecc71", linestyle="--", alpha=0.7, label="A/B boundary (70%)")
    ax2.axhline(y=90, color="#f39c12", linestyle="--", alpha=0.7, label="B/C boundary (90%)")
    ax2.set_ylabel("Cumulative Revenue %", fontsize=12)
    ax2.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "abc_pareto_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Chart 2: Category Distribution Pie ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cat_counts = brand_revenue.groupby("Category")["Brand"].count()
    cat_revenue = brand_revenue.groupby("Category")["Revenue"].sum()

    for ax, data, title in [
        (axes[0], cat_counts, "Brand Count by Category"),
        (axes[1], cat_revenue, "Revenue by Category"),
    ]:
        wedges, texts, autotexts = ax.pie(
            data, labels=data.index,
            colors=[colors.get(c, "#bdc3c7") for c in data.index],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12}
        )
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.suptitle("ABC Category Distribution", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "abc_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[ABC] Charts saved to {output_dir}")