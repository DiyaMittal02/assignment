"""
Additional Insights Module (Task 6)
====================================
Goes beyond the required analyses to deliver extra business intelligence:

    1. Sales Seasonality & Monthly Trends
    2. Store Performance Comparison
    3. Product Classification Mix Analysis
    4. Price Sensitivity / Elasticity Indicators
    5. Dead Stock & Slow-Moving Inventory Detection
    6. Profit Margin Insights (Sales Price vs Purchase Price)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def sales_trends(sales, output_dir=None):
    """
    Analyze monthly revenue trends, seasonality, and day-of-week patterns.

    Returns
    -------
    dict with 'monthly', 'dayofweek', 'classification' DataFrames.
    """
    print("=" * 60)
    print("  SALES TRENDS & SEASONALITY")
    print("=" * 60)

    # Monthly revenue
    monthly = (
        sales.groupby("YearMonth")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("SalesQuantity", "sum"),
            Transactions=("Revenue", "count"),
            AvgPrice=("SalesPrice", "mean"),
        )
        .reset_index()
    )

    # Growth rate
    monthly["Revenue_MoM"] = monthly["Revenue"].pct_change() * 100

    best_month = monthly.loc[monthly["Revenue"].idxmax()]
    worst_month = monthly.loc[monthly["Revenue"].idxmin()]
    print(f"  Best Month       : {best_month['YearMonth']} (${best_month['Revenue']:,.0f})")
    print(f"  Worst Month      : {worst_month['YearMonth']} (${worst_month['Revenue']:,.0f})")
    print(f"  Avg Monthly Rev  : ${monthly['Revenue'].mean():,.0f}")

    # Day of Week patterns
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = sales.groupby("DayOfWeek")["Revenue"].sum().reset_index()
    dow["DayName"] = dow["DayOfWeek"].map(lambda x: dow_names[x] if x < 7 else "?")

    best_day = dow.loc[dow["Revenue"].idxmax()]
    print(f"  Best Sales Day   : {best_day['DayName']} (${best_day['Revenue']:,.0f})")

    # Classification breakdown
    if "Classification" in sales.columns:
        class_rev = (
            sales.groupby("Classification")
            .agg(Revenue=("Revenue", "sum"), Quantity=("SalesQuantity", "sum"))
            .sort_values("Revenue", ascending=False)
            .reset_index()
        )
        print(f"\n  Revenue by Classification:")
        for _, row in class_rev.iterrows():
            pct = row["Revenue"] / class_rev["Revenue"].sum() * 100
            print(f"    {row['Classification']:<20} ${row['Revenue']:>14,.2f} ({pct:.1f}%)")

    print("=" * 60)

    if output_dir:
        _plot_sales_trends(monthly, dow, sales, output_dir)

    return {"monthly": monthly, "dayofweek": dow}


def store_performance(sales, output_dir=None):
    """
    Compare store-level performance on revenue, volume, and pricing.
    """
    print("=" * 60)
    print("  STORE PERFORMANCE COMPARISON")
    print("=" * 60)

    store_df = (
        sales.groupby("Store")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("SalesQuantity", "sum"),
            Transactions=("Revenue", "count"),
            AvgPrice=("SalesPrice", "mean"),
            UniqueBrands=("Brand", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )

    store_df["AvgTransactionValue"] = store_df["Revenue"] / store_df["Transactions"]

    for _, row in store_df.iterrows():
        print(f"  Store {row['Store']}: ${row['Revenue']:>12,.2f} | "
              f"{row['Quantity']:>8,.0f} units | "
              f"Avg Txn ${row['AvgTransactionValue']:.2f}")

    print(f"\n  Total Stores: {len(store_df)}")
    print("=" * 60)

    if output_dir:
        _plot_store_performance(store_df, output_dir)

    return store_df


def dead_stock_analysis(sales, end_inv, output_dir=None):
    """
    Identify slow-moving and dead stock items still in ending inventory
    but with minimal or no sales.
    """
    print("=" * 60)
    print("  DEAD STOCK & SLOW-MOVING INVENTORY")
    print("=" * 60)

    # Brands in ending inventory
    inv_brands = end_inv.groupby("Brand").agg(
        EndingStock=("onHand", "sum"),
        AvgPrice=("Price", "mean"),
    ).reset_index()
    inv_brands["InventoryValue"] = inv_brands["EndingStock"] * inv_brands["AvgPrice"]

    # Total sales per brand
    sales_brands = sales.groupby("Brand")["SalesQuantity"].sum().reset_index()
    sales_brands.columns = ["Brand", "TotalSold"]

    merged = inv_brands.merge(sales_brands, on="Brand", how="left")
    merged["TotalSold"].fillna(0, inplace=True)

    # Stock-to-Sales ratio (higher = slower moving)
    merged["StockToSales"] = np.where(
        merged["TotalSold"] > 0,
        merged["EndingStock"] / merged["TotalSold"],
        np.inf
    )

    # Classify
    def classify_movement(row):
        if row["TotalSold"] == 0:
            return "Dead Stock"
        elif row["StockToSales"] > 2.0:
            return "Slow Moving"
        elif row["StockToSales"] > 0.5:
            return "Normal"
        else:
            return "Fast Moving"

    merged["MovementClass"] = merged.apply(classify_movement, axis=1)

    summary = merged.groupby("MovementClass").agg(
        BrandCount=("Brand", "count"),
        TotalValue=("InventoryValue", "sum"),
    )

    for cls, row in summary.iterrows():
        pct_value = row["TotalValue"] / summary["TotalValue"].sum() * 100
        print(f"  {cls:<15}: {int(row['BrandCount']):>4} brands | "
              f"${row['TotalValue']:>12,.2f} ({pct_value:.1f}% of inv value)")

    dead = merged[merged["MovementClass"] == "Dead Stock"]
    dead_value = dead["InventoryValue"].sum()
    print(f"\n  Total Dead Stock Value    : ${dead_value:,.2f}")
    print(f"  Brands with Zero Sales   : {len(dead)}")
    print("=" * 60)

    if output_dir:
        _plot_dead_stock(merged, output_dir)

    return merged


def profit_margin_analysis(sales, purchase_prices, output_dir=None):
    """
    Analyze profit margins by comparing sales prices to purchase prices.
    """
    print("=" * 60)
    print("  PROFIT MARGIN ANALYSIS")
    print("=" * 60)

    # Average selling price per brand
    avg_sell = (
        sales.groupby("Brand")["SalesPrice"]
        .mean()
        .reset_index()
    )
    avg_sell.columns = ["Brand", "AvgSellPrice"]

    # Average purchase price per brand
    avg_buy = (
        purchase_prices.groupby("Brand")["PurchasePrice"]
        .mean()
        .reset_index()
    )
    avg_buy.columns = ["Brand", "AvgBuyPrice"]

    margin_df = avg_sell.merge(avg_buy, on="Brand", how="inner")
    margin_df["Margin"] = margin_df["AvgSellPrice"] - margin_df["AvgBuyPrice"]
    margin_df["MarginPct"] = (margin_df["Margin"] / margin_df["AvgSellPrice"]) * 100

    # Add total revenue
    brand_rev = sales.groupby("Brand")["Revenue"].sum().reset_index()
    margin_df = margin_df.merge(brand_rev, on="Brand", how="left")

    margin_df = margin_df.sort_values("Revenue", ascending=False).reset_index(drop=True)

    avg_margin = margin_df["MarginPct"].mean()
    high_margin = margin_df[margin_df["MarginPct"] > 50]
    negative_margin = margin_df[margin_df["MarginPct"] < 0]

    print(f"  Brands Analyzed          : {len(margin_df)}")
    print(f"  Average Margin           : {avg_margin:.1f}%")
    print(f"  High Margin (>50%)       : {len(high_margin)} brands")
    print(f"  Negative Margin          : {len(negative_margin)} brands")

    if len(negative_margin) > 0:
        print(f"\n  ⚠ Negative Margin Brands:")
        for _, row in negative_margin.head(10).iterrows():
            print(f"    {row['Brand']}: {row['MarginPct']:.1f}% "
                  f"(sell ${row['AvgSellPrice']:.2f} vs buy ${row['AvgBuyPrice']:.2f})")

    print("=" * 60)

    if output_dir:
        _plot_margins(margin_df, output_dir)

    return margin_df


# ========================= VISUALIZATION FUNCTIONS =========================


def _plot_sales_trends(monthly, dow, sales, output_dir):
    import os

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Monthly Revenue Trend
    ax = axes[0, 0]
    months_str = monthly["YearMonth"].astype(str)
    ax.fill_between(range(len(months_str)), monthly["Revenue"], alpha=0.3, color="#3498db")
    ax.plot(range(len(months_str)), monthly["Revenue"], color="#2c3e50", linewidth=2, marker="o", markersize=4)
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=45, fontsize=7)
    ax.set_ylabel("Revenue ($)", fontsize=11)
    ax.set_title("Monthly Revenue Trend", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Monthly Quantity Trend
    ax = axes[0, 1]
    ax.bar(range(len(months_str)), monthly["Quantity"], color="#27ae60", alpha=0.8)
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=45, fontsize=7)
    ax.set_ylabel("Units Sold", fontsize=11)
    ax.set_title("Monthly Sales Volume", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Day-of-week pattern
    ax = axes[1, 0]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_labels = [dow_names[int(d)] for d in dow["DayOfWeek"]]
    colors_dow = ["#3498db" if d < 5 else "#e74c3c" for d in dow["DayOfWeek"]]
    ax.bar(dow_labels, dow["Revenue"], color=colors_dow, alpha=0.85)
    ax.set_ylabel("Total Revenue ($)", fontsize=11)
    ax.set_title("Revenue by Day of Week", fontsize=13, fontweight="bold")

    # Classification pie
    ax = axes[1, 1]
    if "Classification" in sales.columns:
        class_rev = sales.groupby("Classification")["Revenue"].sum().sort_values(ascending=False)
        class_rev = class_rev.head(8)
        ax.pie(class_rev, labels=class_rev.index, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 9})
        ax.set_title("Revenue by Product Classification", fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Classification data", ha="center", va="center", fontsize=12)

    plt.suptitle("Sales Trends Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sales_trends.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Trends] Charts saved to {output_dir}")


def _plot_store_performance(store_df, output_dir):
    import os

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    stores = store_df["Store"].astype(str)

    ax = axes[0]
    ax.bar(range(len(stores)), store_df["Revenue"], color="#3498db", alpha=0.85)
    ax.set_xticks(range(len(stores)))
    ax.set_xticklabels(stores, rotation=45, fontsize=8)
    ax.set_ylabel("Revenue ($)", fontsize=11)
    ax.set_title("Revenue by Store", fontsize=13, fontweight="bold")

    ax = axes[1]
    ax.bar(range(len(stores)), store_df["Quantity"], color="#27ae60", alpha=0.85)
    ax.set_xticks(range(len(stores)))
    ax.set_xticklabels(stores, rotation=45, fontsize=8)
    ax.set_ylabel("Units Sold", fontsize=11)
    ax.set_title("Volume by Store", fontsize=13, fontweight="bold")

    ax = axes[2]
    ax.bar(range(len(stores)), store_df["AvgTransactionValue"], color="#e67e22", alpha=0.85)
    ax.set_xticks(range(len(stores)))
    ax.set_xticklabels(stores, rotation=45, fontsize=8)
    ax.set_ylabel("Avg Transaction Value ($)", fontsize=11)
    ax.set_title("Avg Transaction Value by Store", fontsize=13, fontweight="bold")

    plt.suptitle("Store Performance Dashboard", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "store_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Store] Charts saved to {output_dir}")


def _plot_dead_stock(merged, output_dir):
    import os

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Movement class distribution
    ax = axes[0]
    class_counts = merged.groupby("MovementClass")["Brand"].count()
    colors_map = {
        "Dead Stock": "#e74c3c", "Slow Moving": "#f39c12",
        "Normal": "#3498db", "Fast Moving": "#27ae60"
    }
    c = [colors_map.get(cls, "#bdc3c7") for cls in class_counts.index]
    ax.pie(class_counts, labels=class_counts.index, colors=c,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax.set_title("Inventory Movement Classification", fontsize=13, fontweight="bold")

    # Value at risk
    ax = axes[1]
    class_value = merged.groupby("MovementClass")["InventoryValue"].sum().sort_values(ascending=False)
    c2 = [colors_map.get(cls, "#bdc3c7") for cls in class_value.index]
    ax.bar(class_value.index, class_value.values, color=c2, alpha=0.85)
    ax.set_ylabel("Inventory Value ($)", fontsize=11)
    ax.set_title("Inventory Value by Movement Class", fontsize=13, fontweight="bold")
    ax.tick_params(axis='x', rotation=20)

    plt.suptitle("Dead Stock & Slow-Moving Inventory", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dead_stock_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DeadStock] Charts saved to {output_dir}")


def _plot_margins(margin_df, output_dir):
    import os

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 20 brands by revenue with margin overlay
    top = margin_df.head(20)
    ax = axes[0]
    colors = np.where(top["MarginPct"] >= 0, "#27ae60", "#e74c3c")
    ax.barh(range(len(top)), top["MarginPct"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Brand"], fontsize=8)
    ax.set_xlabel("Margin %", fontsize=11)
    ax.set_title("Profit Margin - Top 20 Brands by Revenue", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    # Margin distribution
    ax = axes[1]
    ax.hist(margin_df["MarginPct"].dropna(), bins=40, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(margin_df["MarginPct"].mean(), color="#e74c3c", linestyle="--",
               linewidth=2, label=f"Mean ({margin_df['MarginPct'].mean():.1f}%)")
    ax.set_xlabel("Margin %", fontsize=11)
    ax.set_ylabel("Number of Brands", fontsize=11)
    ax.set_title("Margin Distribution Across All Brands", fontsize=13, fontweight="bold")
    ax.legend()

    plt.suptitle("Profit Margin Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_margin_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Margin] Charts saved to {output_dir}")
