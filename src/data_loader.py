"""
Data Loader Module
==================
Handles loading and initial validation of all datasets for the
Wine & Spirits inventory optimization project.

Datasets:
    - SalesFINAL12312016.csv         : Transaction-level sales records
    - PurchasesFINAL12312016.csv     : Detailed purchase/procurement records
    - BegInvFINAL12312016.csv        : Beginning inventory snapshot
    - EndInvFINAL12312016.csv        : Ending inventory snapshot
    - InvoicePurchases12312016.csv   : Invoice-level purchase summaries
    - 2017PurchasePricesDec.csv      : Product pricing reference data
"""

import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data(sample_frac=None):
    """
    Load all six datasets from the data directory.

    Parameters
    ----------
    sample_frac : float, optional
        If provided, randomly sample this fraction of rows from the two
        largest files (Sales, Purchases) to speed up development runs.
        Example: sample_frac=0.1 loads 10 % of rows.

    Returns
    -------
    tuple of DataFrames
        (sales, purchases, beg_inv, end_inv, invoice_purchases, purchase_prices)
    """

    print("[DataLoader] Loading datasets ...")

    sales = pd.read_csv(os.path.join(DATA_DIR, "SalesFINAL12312016.csv"))
    purchases = pd.read_csv(os.path.join(DATA_DIR, "PurchasesFINAL12312016.csv"))
    beg_inv = pd.read_csv(os.path.join(DATA_DIR, "BegInvFINAL12312016.csv"))
    end_inv = pd.read_csv(os.path.join(DATA_DIR, "EndInvFINAL12312016.csv"))
    invoice_purchases = pd.read_csv(os.path.join(DATA_DIR, "InvoicePurchases12312016.csv"))
    purchase_prices = pd.read_csv(os.path.join(DATA_DIR, "2017PurchasePricesDec.csv"))

    if sample_frac is not None and 0 < sample_frac < 1:
        sales = sales.sample(frac=sample_frac, random_state=42)
        purchases = purchases.sample(frac=sample_frac, random_state=42)
        print(f"[DataLoader] Sampled {sample_frac*100:.0f}% of Sales & Purchases.")

    print(f"[DataLoader]   Sales         : {len(sales):>10,} rows")
    print(f"[DataLoader]   Purchases     : {len(purchases):>10,} rows")
    print(f"[DataLoader]   Beg Inventory : {len(beg_inv):>10,} rows")
    print(f"[DataLoader]   End Inventory : {len(end_inv):>10,} rows")
    print(f"[DataLoader]   Invoices      : {len(invoice_purchases):>10,} rows")
    print(f"[DataLoader]   Price List    : {len(purchase_prices):>10,} rows")
    print("[DataLoader] All datasets loaded successfully.\n")

    return sales, purchases, beg_inv, end_inv, invoice_purchases, purchase_prices