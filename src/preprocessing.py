"""
Preprocessing Module
====================
Cleans and enriches raw dataframes so downstream analyses can work with
consistent, well-typed columns.

Key transformations:
    - Strip whitespace from column names
    - Parse date columns to datetime
    - Derive Revenue, Profit Margin, Lead Time
    - Report missing-value statistics
"""

import pandas as pd
import numpy as np


def _report_quality(df, name):
    """Print a concise data-quality summary."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"[Preprocessing] {name} - missing values:")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"                  {col}: {cnt:,} ({pct:.1f}%)")
    else:
        print(f"[Preprocessing] {name} - no missing values detected.")


def preprocess_sales(sales):
    """
    Clean and enrich the Sales dataframe.

    Derived columns
    ----------------
    - Revenue   : alias for SalesDollars
    - YearMonth : period column for time-series grouping
    - Month     : calendar month (1-12)
    - DayOfWeek : 0=Mon ... 6=Sun
    """
    sales = sales.copy()
    sales.columns = sales.columns.str.strip()

    sales["SalesDate"] = pd.to_datetime(sales["SalesDate"], errors="coerce")

    # Revenue alias
    sales["Revenue"] = sales["SalesDollars"]

    # Time features
    sales["YearMonth"] = sales["SalesDate"].dt.to_period("M")
    sales["Month"] = sales["SalesDate"].dt.month
    sales["DayOfWeek"] = sales["SalesDate"].dt.dayofweek

    # Profit margin where both columns exist
    if {"SalesPrice", "Volume"}.issubset(sales.columns):
        sales["UnitRevenue"] = sales["SalesPrice"]

    _report_quality(sales, "Sales")
    print(f"[Preprocessing] Sales date range: "
          f"{sales['SalesDate'].min().date()} to {sales['SalesDate'].max().date()}\n")

    return sales


def preprocess_purchases(purchases):
    """
    Clean and enrich the Purchases dataframe.

    Derived columns
    ----------------
    - LeadTime    : days between PO and Receiving
    - YearMonth   : period for time-series grouping
    - UnitCost    : per-unit purchase cost (Dollars / Quantity)
    """
    purchases = purchases.copy()
    purchases.columns = purchases.columns.str.strip()

    purchases["PODate"] = pd.to_datetime(purchases["PODate"], errors="coerce")
    purchases["ReceivingDate"] = pd.to_datetime(purchases["ReceivingDate"], errors="coerce")
    purchases["InvoiceDate"] = pd.to_datetime(purchases["InvoiceDate"], errors="coerce")
    purchases["PayDate"] = pd.to_datetime(purchases["PayDate"], errors="coerce")

    # Lead Time
    purchases["LeadTime"] = (
        purchases["ReceivingDate"] - purchases["PODate"]
    ).dt.days

    # Negative or extreme lead times -> NaN
    purchases.loc[purchases["LeadTime"] < 0, "LeadTime"] = np.nan
    purchases.loc[purchases["LeadTime"] > 365, "LeadTime"] = np.nan

    # Time features
    purchases["YearMonth"] = purchases["PODate"].dt.to_period("M")

    # Unit cost
    purchases["UnitCost"] = np.where(
        purchases["Quantity"] > 0,
        purchases["Dollars"] / purchases["Quantity"],
        np.nan
    )

    _report_quality(purchases, "Purchases")
    print(f"[Preprocessing] Purchases date range: "
          f"{purchases['PODate'].min().date()} to {purchases['PODate'].max().date()}\n")

    return purchases