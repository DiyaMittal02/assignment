"""
Inventory Turnover Analysis Module
====================================
Measures how efficiently inventory is sold and replaced over the period.

    Turnover Ratio  = Cost of Goods Sold / Average Inventory Value
    Days Sales of Inventory (DSI) = 365 / Turnover Ratio

Provides both aggregate and store-level turnover metrics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inventory_turnover(sales, beg_inv, end_inv, output_dir=None):
    """
    Calculate inventory turnover ratio and days-sales-of-inventory.

    Parameters
    ----------
    sales : pd.DataFrame
        Preprocessed sales data.
    beg_inv : pd.DataFrame
        Beginning inventory snapshot.
    end_inv : pd.DataFrame
        Ending inventory snapshot.
    output_dir : str, optional
        Directory to save visualizations.

    Returns
    -------
    pd.DataFrame
        Store-level turnover analysis plus overall metrics.
    """
    # ---- Overall turnover ----
    cogs = sales["SalesDollars"].sum()
    beg_val = (beg_inv["onHand"] * beg_inv["Price"]).sum()
    end_val = (end_inv["onHand"] * end_inv["Price"]).sum()
    avg_inv_value = (beg_val + end_val) / 2

    overall_turnover = cogs / avg_inv_value if avg_inv_value > 0 else 0
    overall_dsi = 365 / overall_turnover if overall_turnover > 0 else np.inf

    print("=" * 60)
    print("  INVENTORY TURNOVER ANALYSIS")
    print("=" * 60)
    print(f"  Cost of Goods Sold       : ${cogs:>14,.2f}")
    print(f"  Beginning Inventory Val  : ${beg_val:>14,.2f}")
    print(f"  Ending Inventory Val     : ${end_val:>14,.2f}")
    print(f"  Average Inventory Val    : ${avg_inv_value:>14,.2f}")
    print(f"  Overall Turnover Ratio   : {overall_turnover:.2f}x")
    print(f"  Days Sales of Inventory  : {overall_dsi:.1f} days")
    print("=" * 60)

    # ---- Store-level turnover ----
    store_sales = sales.groupby("Store")["SalesDollars"].sum().reset_index()
    store_sales.columns = ["Store", "COGS"]

    store_beg = beg_inv.copy()
    store_beg["InvValue"] = store_beg["onHand"] * store_beg["Price"]
    store_beg_val = store_beg.groupby("Store")["InvValue"].sum().reset_index()
    store_beg_val.columns = ["Store", "BegInvValue"]

    store_end = end_inv.copy()
    store_end["InvValue"] = store_end["onHand"] * store_end["Price"]
    store_end_val = store_end.groupby("Store")["InvValue"].sum().reset_index()
    store_end_val.columns = ["Store", "EndInvValue"]

    store_df = store_sales.merge(store_beg_val, on="Store", how="outer")
    store_df = store_df.merge(store_end_val, on="Store", how="outer")
    store_df.fillna(0, inplace=True)

    store_df["AvgInvValue"] = (store_df["BegInvValue"] + store_df["EndInvValue"]) / 2
    store_df["TurnoverRatio"] = np.where(
        store_df["AvgInvValue"] > 0,
        store_df["COGS"] / store_df["AvgInvValue"],
        0
    )
    store_df["DSI"] = np.where(
        store_df["TurnoverRatio"] > 0,
        365 / store_df["TurnoverRatio"],
        np.nan
    )
    store_df = store_df.sort_values("TurnoverRatio", ascending=False).reset_index(drop=True)

    if output_dir:
        _plot_turnover(store_df, overall_turnover, overall_dsi, output_dir)

    return store_df


def _plot_turnover(store_df, overall_turnover, overall_dsi, output_dir):
    """Generate inventory turnover visualizations."""
    import os

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Chart 1: Turnover ratio by store
    ax = axes[0]
    stores = store_df["Store"].astype(str)
    colors = np.where(store_df["TurnoverRatio"] >= overall_turnover, "#27ae60", "#e74c3c")
    ax.bar(range(len(stores)), store_df["TurnoverRatio"], color=colors, alpha=0.8)
    ax.axhline(y=overall_turnover, color="#2c3e50", linestyle="--", linewidth=2,
               label=f"Overall Avg ({overall_turnover:.2f}x)")
    ax.set_xticks(range(len(stores)))
    ax.set_xticklabels(stores, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Turnover Ratio", fontsize=11)
    ax.set_title("Inventory Turnover by Store", fontsize=13, fontweight="bold")
    ax.legend()

    # Chart 2: DSI by store
    ax = axes[1]
    valid = store_df.dropna(subset=["DSI"])
    colors2 = np.where(valid["DSI"] <= overall_dsi, "#27ae60", "#e74c3c")
    ax.bar(range(len(valid)), valid["DSI"], color=colors2, alpha=0.8)
    ax.axhline(y=overall_dsi, color="#2c3e50", linestyle="--", linewidth=2,
               label=f"Overall Avg ({overall_dsi:.1f} days)")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid["Store"].astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Days Sales of Inventory", fontsize=11)
    ax.set_title("Days of Inventory by Store", fontsize=13, fontweight="bold")
    ax.legend()

    plt.suptitle("Inventory Turnover Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inventory_turnover.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Turnover] Charts saved to {output_dir}")