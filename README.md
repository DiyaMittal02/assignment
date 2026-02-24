# 🍷 Wine & Spirits Inventory Optimization Suite

A comprehensive data-driven analysis pipeline for a multi-location retail wine & spirits company. This project processes millions of transaction records across sales, purchases, and inventory datasets to optimize inventory control, reduce inefficiencies, and extract actionable business insights.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Project Architecture](#project-architecture)
- [Analyses Performed](#analyses-performed)
- [Key Findings & Insights](#key-findings--insights)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)

---

## 🎯 Project Overview

Traditional spreadsheet-based analysis falls short when dealing with millions of retail transactions. This project implements a modular, scalable Python pipeline that performs six core analyses to help the company make data-informed decisions about inventory management, procurement strategy, and sales optimization.

---

## 🎯 Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | **Inventory Optimization** - Determine ideal inventory levels for different product categories | ✅ |
| 2 | **Sales & Purchase Insights** - Identify trends, top-performing products, and supplier efficiency | ✅ |
| 3 | **Process Improvement** - Optimize procurement and stock control to minimize financial loss | ✅ |

---

## 📊 Dataset Description

The project utilizes six interconnected datasets from a retail wine & spirits operation:

| File | Description | Key Columns |
|------|-------------|-------------|
| `SalesFINAL12312016.csv` | Transaction-level sales records (~2.6M rows) | Brand, SalesQuantity, SalesDollars, SalesDate, Store |
| `PurchasesFINAL12312016.csv` | Detailed purchase/procurement records (~8.7M rows) | Brand, VendorName, PODate, ReceivingDate, Quantity, Dollars |
| `BegInvFINAL12312016.csv` | Beginning inventory snapshot | Brand, Store, onHand, Price |
| `EndInvFINAL12312016.csv` | Ending inventory snapshot | Brand, Store, onHand, Price |
| `InvoicePurchases12312016.csv` | Invoice-level purchase summaries | VendorName, InvoiceDate, PayDate, Dollars |
| `2017PurchasePricesDec.csv` | Product pricing reference | Brand, PurchasePrice, Price, Classification |

> **Note:** Dataset files are not included in the repository due to size (~570 MB total). Download from the [Kaggle Dataset](https://www.kaggle.com/datasets/sloozecareers/slooze-challenge/data) and place in the `data/` directory.

---

## 🏗️ Project Architecture

```
Snooze-Challenge/
├── main.py                         # Main orchestration script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── data/                           # Dataset directory (not in git)
│   ├── SalesFINAL12312016.csv
│   ├── PurchasesFINAL12312016.csv
│   ├── BegInvFINAL12312016.csv
│   ├── EndInvFINAL12312016.csv
│   ├── InvoicePurchases12312016.csv
│   └── 2017PurchasePricesDec.csv
├── src/                            # Analysis modules
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & validation
│   ├── preprocessing.py            # Data cleaning & enrichment
│   ├── abc_analysis.py             # Task 2: ABC Classification
│   ├── demand_forecasting.py       # Task 1: SARIMAX Forecasting
│   ├── eoq.py                      # Task 3: Economic Order Quantity
│   ├── reorder_point.py            # Task 4: Reorder Point Analysis
│   ├── lead_time_analysis.py       # Task 5: Lead Time Analysis
│   ├── supplier_analysis.py        # Supplier Performance Scoring
│   ├── inventory_turnover.py       # Inventory Turnover & DSI
│   └── additional_insights.py      # Task 6: Extra Business Intelligence
└── output/                         # Generated reports & charts
    ├── *.csv                       # Data export files
    └── *.png                       # Visualization charts
```

---

## 📈 Analyses Performed

### Task 1: Demand Forecasting
- **Method:** SARIMAX(1,1,1) time-series model
- **Frequency:** Weekly aggregation (optimized for the dataset's ~2-month span)
- **Output:** 4-week ahead forecasts with 95% confidence intervals
- **Metrics:** MAE, RMSE, AIC for model evaluation
- **Scope:** Top 10 brands by sales volume

### Task 2: ABC Analysis (Pareto Classification)
- **Method:** Revenue-based cumulative percentage classification
- **Categories:**
  - **A (Vital Few):** Top 70% of revenue
  - **B (Moderate):** Next 20% (70-90%)
  - **C (Trivial Many):** Bottom 10% (90-100%)
- **Output:** Pareto chart + category distribution pie charts

### Task 3: Economic Order Quantity (EOQ)
- **Formula:** Wilson EOQ = sqrt(2DS / H)
- **Parameters:**
  - Ordering cost: $150/order
  - Holding cost: 20% of average purchase price/year
- **Output:** Optimal order quantities, orders/year, total inventory cost

### Task 4: Reorder Point Analysis
- **Formula:** ROP = (Avg Daily Demand x Avg Lead Time) + Safety Stock
- **Safety Stock:** Z x sqrt(LT x sigma_d^2 + d_avg^2 x sigma_LT^2) at 95% service level (Z=1.65)
- **Output:** Per-brand reorder points with safety stock components

### Task 5: Lead Time Analysis
- **Metrics:** Distribution, trend over time, vendor reliability scoring
- **Bottleneck Detection:** Identifies suppliers above 90th percentile lead time
- **Payment Cycle:** Invoice-to-pay analysis
- **Output:** 4-panel dashboard (distribution, trends, slowest suppliers, box plots)

### Task 6: Additional Insights
- **Sales Trends & Seasonality:** Monthly revenue trends, day-of-week patterns
- **Store Performance:** Revenue, volume, and transaction value comparison across locations
- **Dead Stock Detection:** Identifies zero-sales and slow-moving inventory
- **Profit Margin Analysis:** Sales price vs purchase price margin analysis per brand

---

## 🔑 Key Findings & Insights

### Inventory Optimization
- **74.5% of inventory** is classified as slow-moving or dead stock, representing significant capital lockup
- Dead stock alone accounts for 31.3% of brands in ending inventory
- The ABC analysis reveals a classic Pareto distribution - a small number of A-category brands drive the majority of revenue

### Supply Chain Efficiency
- **Average lead time: 7.6 days** (median: 8 days) with reasonable consistency
- Bottleneck suppliers identified with lead times exceeding the 90th percentile
- Lead time trend shows monthly variation, suggesting room for supplier negotiation

### Sales Performance
- **Friday and Saturday** are peak sales days, contributing the highest weekly revenue
- Product Classification 1 dominates at 62% of revenue vs 38% for Classification 2
- Significant revenue variation across stores highlights opportunities for location-specific strategies

### Procurement Recommendations
1. **Reduce dead stock** - Implement clearance strategies for 31% of brands with zero sales
2. **Negotiate lead times** with bottleneck suppliers identified in the analysis
3. **Implement EOQ ordering** - Calculated optimal order quantities reduce total inventory cost
4. **Set reorder points** with safety stock at 95% service level to prevent stockouts
5. **Focus on A-category brands** - Prioritize inventory management for high-revenue products

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Snooze-Challenge.git
cd Snooze-Challenge

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
1. Download the dataset from the [provided source](https://www.kaggle.com/)
2. Place all 6 CSV files in the `data/` directory

---

## 🚀 How to Run

### Option 1: Interactive Dashboard (Recommended)

```bash
# Launch the Streamlit dashboard
streamlit run dashboard.py
```

This opens a **professional interactive dashboard** at `http://localhost:8501` with:
- 📊 9 analysis pages with sidebar navigation
- 🎨 Interactive Plotly charts (zoom, hover, filter)
- 📋 KPI cards with gradient styling
- 🔍 Data explorer with raw data access
- 📈 Real-time brand-level demand forecasting

### Option 2: CLI Pipeline (Batch Export)

```bash
# Run the complete analysis pipeline
python main.py
```

The script will:
1. Load and validate all 6 datasets
2. Preprocess and enrich the data
3. Execute all 6 analysis tasks sequentially
4. Generate CSV reports and PNG visualizations
5. Save all outputs to the `output/` directory

**Expected runtime:** ~3-5 minutes (depending on hardware)

---

## 📁 Output Files

### CSV Reports
| File | Description |
|------|-------------|
| `abc_analysis.csv` | Brand-level ABC classification with revenue stats |
| `eoq_analysis.csv` | EOQ, orders/year, and cost breakdown per brand |
| `reorder_point_analysis.csv` | Reorder points and safety stock per brand |
| `supplier_performance.csv` | Supplier scorecard with delivery metrics |
| `inventory_turnover.csv` | Store-level turnover ratios and DSI |
| `lead_time_vendor.csv` | Vendor-level lead time statistics |
| `profit_margins.csv` | Sales vs purchase price margin analysis |
| `dead_stock_analysis.csv` | Inventory movement classification |
| `store_performance.csv` | Store-level sales performance metrics |
| `forecast_metrics.csv` | SARIMAX model evaluation metrics |

### Visualizations (PNG Charts)
| File | Description |
|------|-------------|
| `abc_pareto_chart.png` | Pareto bar chart with cumulative revenue line |
| `abc_distribution.png` | ABC category pie charts |
| `demand_forecasts.png` | SARIMAX forecasts with confidence intervals |
| `eoq_analysis.png` | EOQ scatter plot and top brands |
| `eoq_cost_breakdown.png` | Ordering vs holding cost comparison |
| `reorder_point_analysis.png` | ROP components and safety stock analysis |
| `supplier_performance.png` | 4-panel supplier dashboard |
| `inventory_turnover.png` | Store-level turnover comparison |
| `lead_time_analysis.png` | 4-panel lead time dashboard |
| `sales_trends.png` | Revenue trends, day patterns, classifications |
| `store_performance.png` | Store comparison dashboard |
| `dead_stock_analysis.png` | Inventory movement classification |
| `profit_margin_analysis.png` | Margin distribution and brand-level analysis |

---

## 🔬 Methodology

### Data Preprocessing
- Column name standardization (whitespace stripping)
- Date parsing with error handling
- Lead time calculation with outlier filtering (negative or >365 days)
- Feature engineering: YearMonth, DayOfWeek, UnitCost, Revenue

### Statistical Models
- **SARIMAX(1,1,1):** Seasonal ARIMA with exogenous variables for demand forecasting
- **Wilson EOQ Model:** Classical inventory optimization formula with actual purchase prices
- **Service Level Safety Stock:** Z-score approach (95% service level, Z=1.65) incorporating both demand and lead time variability

### Data Quality
- Missing value detection and reporting
- Negative lead time filtering
- Minimum data point thresholds for forecasting (4+ weeks of non-zero sales)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **Streamlit** | Interactive dashboard framework |
| **Plotly** | Dynamic charting and visualizations |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Static chart generation |
| **Seaborn** | Statistical data visualization |
| **Statsmodels** | SARIMAX time-series modeling |
| **scikit-learn** | Statistical utilities |

---

## 📄 License

This project is developed as part of an inventory optimization challenge.

---

*Built with Python - Powered by Data Science*
