"""
=============================================================================
  Wine & Spirits Inventory Optimization Suite
  ============================================
  A comprehensive data-driven analysis pipeline for a multi-location retail
  wine & spirits company. This project addresses inventory optimization,
  demand forecasting, and business intelligence extraction.

  Tasks Covered:
    1. Demand Forecasting          (SARIMAX time-series)
    2. ABC Analysis                (Pareto classification)
    3. Economic Order Quantity      (Wilson EOQ model)
    4. Reorder Point Analysis       (Safety stock + service level)
    5. Lead Time Analysis           (Supply chain efficiency)
    6. Additional Insights          (Trends, dead stock, margins, stores)

  Author  : Inventory Analytics Team
  Dataset : Retail Wine & Spirits (6 CSV files)
=============================================================================
"""

import os
import sys
import time

# ---- Configure paths ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Module imports ----
from src.data_loader import load_data
from src.preprocessing import preprocess_sales, preprocess_purchases
from src.abc_analysis import abc_analysis
from src.eoq import calculate_eoq
from src.reorder_point import reorder_point
from src.supplier_analysis import supplier_performance
from src.inventory_turnover import inventory_turnover
from src.demand_forecasting import forecast_top_brands
from src.lead_time_analysis import lead_time_analysis
from src.additional_insights import (
    sales_trends,
    store_performance,
    dead_stock_analysis,
    profit_margin_analysis
)


def print_banner():
    """Print a professional banner."""
    print("\n" + "=" * 70)
    print("  🍷  WINE & SPIRITS INVENTORY OPTIMIZATION SUITE  🍷")
    print("  " + "-" * 66)
    print("  Comprehensive Inventory Analysis  |  Data-Driven Optimization")
    print("=" * 70 + "\n")


def main():
    start = time.time()
    print_banner()

    # ===== STEP 1: Load Data =====
    sales, purchases, beg_inv, end_inv, invoice_purchases, purchase_prices = load_data()

    # ===== STEP 2: Preprocess =====
    print("\n" + "=" * 70)
    print("  STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    sales = preprocess_sales(sales)
    purchases = preprocess_purchases(purchases)

    # ===== TASK 1: Demand Forecasting =====
    print("\n" + "=" * 70)
    print("  TASK 1: DEMAND FORECASTING")
    print("=" * 70)
    forecasts = forecast_top_brands(sales, n_brands=10, forecast_steps=4,
                                     output_dir=OUTPUT_DIR)

    if forecasts:
        print(f"\n  Successfully forecasted {len(forecasts)} brands.")
        print("  Sample Forecast (next 4 weeks):")
        for brand, data in list(forecasts.items())[:3]:
            fc = data["forecast"]
            print(f"\n  {brand}:")
            for idx, row in fc.iterrows():
                print(f"    Week of {idx.date()}: {row['Predicted']:,.0f} units "
                      f"[{row['Lower_CI']:,.0f} - {row['Upper_CI']:,.0f}]")
    else:
        print("\n  No brands had sufficient data for SARIMAX forecasting.")

    # ===== TASK 2: ABC Analysis =====
    print("\n" + "=" * 70)
    print("  TASK 2: ABC ANALYSIS")
    print("=" * 70)
    abc_df = abc_analysis(sales, output_dir=OUTPUT_DIR)
    print(f"\n  Top 10 A-Category Brands:")
    a_brands = abc_df[abc_df["Category"] == "A"].head(10)
    for _, row in a_brands.iterrows():
        print(f"    {row['Brand']:<30} ${row['Revenue']:>14,.2f} ({row['Revenue %']:.2f}%)")

    # ===== TASK 3: EOQ Analysis =====
    print("\n" + "=" * 70)
    print("  TASK 3: ECONOMIC ORDER QUANTITY (EOQ)")
    print("=" * 70)
    eoq_df = calculate_eoq(sales, purchase_prices=purchase_prices,
                            output_dir=OUTPUT_DIR)
    print(f"\n  Top 10 Brands by EOQ:")
    for _, row in eoq_df.head(10).iterrows():
        print(f"    {row['Brand']:<30} EOQ={row['EOQ']:>8,} units | "
              f"Orders/Year={row['OrdersPerYear']:>3} | "
              f"Total Cost=${row['TotalInventoryCost']:>10,.2f}")

    # ===== TASK 4: Reorder Point Analysis =====
    print("\n" + "=" * 70)
    print("  TASK 4: REORDER POINT ANALYSIS")
    print("=" * 70)
    rop_df = reorder_point(sales, purchases, output_dir=OUTPUT_DIR)
    print(f"\n  Top 10 Brands by Reorder Point:")
    for _, row in rop_df.head(10).iterrows():
        print(f"    {row['Brand']:<30} ROP={row['ReorderPoint']:>8,} units | "
              f"Safety Stock={row['SafetyStock']:>6,}")

    # ===== TASK 5: Lead Time Analysis =====
    print("\n" + "=" * 70)
    print("  TASK 5: LEAD TIME ANALYSIS")
    print("=" * 70)
    lt_results = lead_time_analysis(purchases, output_dir=OUTPUT_DIR)

    # ===== TASK 5 (cont): Supplier Performance =====
    print("\n" + "=" * 70)
    print("  SUPPLIER PERFORMANCE ANALYSIS")
    print("=" * 70)
    supplier_df = supplier_performance(purchases, output_dir=OUTPUT_DIR)
    print(f"\n  Top 10 Suppliers by Spend:")
    for _, row in supplier_df.head(10).iterrows():
        print(f"    {row['VendorName']:<30} ${row['Total_Spent']:>14,.2f} | "
              f"Lead Time: {row['Avg_LeadTime']:.1f} days")

    # ===== TASK 5 (cont): Inventory Turnover =====
    print("\n" + "=" * 70)
    print("  INVENTORY TURNOVER ANALYSIS")
    print("=" * 70)
    turnover_df = inventory_turnover(sales, beg_inv, end_inv, output_dir=OUTPUT_DIR)

    # ===== TASK 6: Additional Insights =====
    print("\n" + "=" * 70)
    print("  TASK 6: ADDITIONAL INSIGHTS & TRENDS")
    print("=" * 70)

    # 6a. Sales Trends
    trends = sales_trends(sales, output_dir=OUTPUT_DIR)

    # 6b. Store Performance
    store_df = store_performance(sales, output_dir=OUTPUT_DIR)

    # 6c. Dead Stock Analysis
    dead_df = dead_stock_analysis(sales, end_inv, output_dir=OUTPUT_DIR)

    # 6d. Profit Margin Analysis
    margin_df = profit_margin_analysis(sales, purchase_prices, output_dir=OUTPUT_DIR)

    # ===== EXPORT RESULTS =====
    print("\n" + "=" * 70)
    print("  EXPORTING RESULTS")
    print("=" * 70)

    abc_df.to_csv(os.path.join(OUTPUT_DIR, "abc_analysis.csv"), index=False)
    eoq_df.to_csv(os.path.join(OUTPUT_DIR, "eoq_analysis.csv"), index=False)
    rop_df.to_csv(os.path.join(OUTPUT_DIR, "reorder_point_analysis.csv"), index=False)
    supplier_df.to_csv(os.path.join(OUTPUT_DIR, "supplier_performance.csv"), index=False)
    turnover_df.to_csv(os.path.join(OUTPUT_DIR, "inventory_turnover.csv"), index=False)
    margin_df.to_csv(os.path.join(OUTPUT_DIR, "profit_margins.csv"), index=False)
    dead_df.to_csv(os.path.join(OUTPUT_DIR, "dead_stock_analysis.csv"), index=False)
    store_df.to_csv(os.path.join(OUTPUT_DIR, "store_performance.csv"), index=False)

    lt_results["vendor_stats"].to_csv(
        os.path.join(OUTPUT_DIR, "lead_time_vendor.csv"), index=False
    )

    print(f"  CSV reports saved to: {OUTPUT_DIR}")

    # ===== COMPLETION =====
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"  ✅  ANALYSIS COMPLETE  |  Total Execution Time: {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  Output Directory: {OUTPUT_DIR}")
    print("  Generated files:")

    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {'📊' if f.endswith('.png') else '📄'} {f} ({size_kb:.1f} KB)")

    print("\n  🍷 Thank you for using the Wine & Spirits Inventory Optimization Suite!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()