"""
Economic Order Quantity (EOQ) Module
====================================
Implements the Wilson EOQ formula to determine the optimal order quantity
that minimizes total inventory costs (ordering + holding costs).

    EOQ = sqrt( (2 * D * S) / H )

Where:
    D = Annual demand (units)
    S = Fixed cost per order ($)
    H = Annual holding cost per unit ($)

Industry Assumptions for Wine & Spirits:
    - Ordering cost: $150 per order (admin, shipping, inspection)
    - Holding cost rate: 20% of average unit purchase price per year
      (storage, insurance, spoilage, capital cost)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDERING_COST = 150       # $ per order placed
HOLDING_COST_RATE = 0.20  # 20% of unit price per year


def calculate_eoq(sales, purchase_prices=None, output_dir=None):
    """
    Calculate EOQ for each brand.

    Parameters
    ----------
    sales : pd.DataFrame
        Preprocessed sales data.
    purchase_prices : pd.DataFrame, optional
        Price list to derive per-unit holding cost. If None, uses a flat $10
        per-unit holding cost assumption.
    output_dir : str, optional
        Directory to save visualizations.

    Returns
    -------
    pd.DataFrame
        EOQ results with annual demand, costs, and optimal order quantities.
    """
    annual_demand = sales.groupby("Brand")["SalesQuantity"].sum().reset_index()
    annual_demand.columns = ["Brand", "AnnualDemand"]

    # Merge actual prices if available
    if purchase_prices is not None:
        avg_price = (
            purchase_prices.groupby("Brand")["PurchasePrice"]
            .mean()
            .reset_index()
        )
        avg_price.columns = ["Brand", "AvgPurchasePrice"]
        eoq_df = annual_demand.merge(avg_price, on="Brand", how="left")
        eoq_df["AvgPurchasePrice"].fillna(eoq_df["AvgPurchasePrice"].median(), inplace=True)
        eoq_df["HoldingCost"] = eoq_df["AvgPurchasePrice"] * HOLDING_COST_RATE
    else:
        eoq_df = annual_demand.copy()
        eoq_df["AvgPurchasePrice"] = np.nan
        eoq_df["HoldingCost"] = 10.0  # flat assumption

    eoq_df["OrderingCost"] = ORDERING_COST

    # EOQ Formula
    eoq_df["EOQ"] = np.sqrt(
        (2 * eoq_df["AnnualDemand"] * eoq_df["OrderingCost"]) / eoq_df["HoldingCost"]
    )
    eoq_df["EOQ"] = eoq_df["EOQ"].round(0).astype(int)

    # Optimal number of orders per year
    eoq_df["OrdersPerYear"] = np.ceil(eoq_df["AnnualDemand"] / eoq_df["EOQ"]).astype(int)

    # Total annual cost at EOQ
    eoq_df["TotalOrderingCost"] = eoq_df["OrdersPerYear"] * ORDERING_COST
    eoq_df["TotalHoldingCost"] = (eoq_df["EOQ"] / 2) * eoq_df["HoldingCost"]
    eoq_df["TotalInventoryCost"] = eoq_df["TotalOrderingCost"] + eoq_df["TotalHoldingCost"]

    eoq_df = eoq_df.sort_values("AnnualDemand", ascending=False).reset_index(drop=True)

    # Print summary
    print("=" * 60)
    print("  ECONOMIC ORDER QUANTITY (EOQ) ANALYSIS")
    print("=" * 60)
    print(f"  Ordering Cost Assumption : ${ORDERING_COST}")
    print(f"  Holding Cost Rate        : {HOLDING_COST_RATE*100:.0f}% of unit price")
    print(f"  Total Brands Analyzed    : {len(eoq_df)}")
    print(f"  Avg EOQ                  : {eoq_df['EOQ'].mean():.0f} units")
    print(f"  Total Inventory Cost     : ${eoq_df['TotalInventoryCost'].sum():,.2f}")
    print("=" * 60)

    if output_dir:
        _plot_eoq(eoq_df, output_dir)

    return eoq_df


def _plot_eoq(eoq_df, output_dir):
    """Generate EOQ visualizations."""
    import os

    top = eoq_df.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: EOQ vs Demand scatter
    ax = axes[0]
    scatter = ax.scatter(
        eoq_df["AnnualDemand"], eoq_df["EOQ"],
        c=eoq_df["TotalInventoryCost"], cmap="YlOrRd",
        alpha=0.7, edgecolors="white", linewidth=0.5, s=50
    )
    ax.set_xlabel("Annual Demand (units)", fontsize=11)
    ax.set_ylabel("EOQ (units)", fontsize=11)
    ax.set_title("EOQ vs Annual Demand", fontsize=13, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Total Inventory Cost ($)")

    # Chart 2: Top 20 brands by EOQ
    ax = axes[1]
    bars = ax.barh(range(len(top)), top["EOQ"], color="#3498db", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Brand"], fontsize=8)
    ax.set_xlabel("Economic Order Quantity (units)", fontsize=11)
    ax.set_title("Top 20 Brands by EOQ", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eoq_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 3: EOQ Cost Breakdown for top brands
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top))
    width = 0.35
    ax.bar(x - width/2, top["TotalOrderingCost"], width, label="Ordering Cost", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, top["TotalHoldingCost"], width, label="Holding Cost", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top["Brand"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cost ($)", fontsize=11)
    ax.set_title("EOQ Cost Breakdown - Top 20 Brands", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eoq_cost_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[EOQ] Charts saved to {output_dir}")