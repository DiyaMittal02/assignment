"""
Wine & Spirits Inventory Optimization - Interactive Dashboard
==============================================================
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, os

warnings.filterwarnings("ignore")

# ────────────────────────────── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Wine & Spirits Inventory Suite",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────── CUSTOM CSS ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 1.2rem 1.5rem; text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.25); color: white;
}
.kpi-card h3 { font-size: 0.85rem; font-weight: 400; margin: 0; opacity: 0.85; }
.kpi-card h1 { font-size: 1.8rem; font-weight: 700; margin: 4px 0 0 0; }
.kpi-blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
.kpi-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.kpi-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
.kpi-red { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
.kpi-purple { background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%); }
.kpi-teal { background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); }
.section-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.6rem; font-weight: 700; margin-bottom: 0.5rem;
}
div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 10px; padding: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #f0f2f6; border-radius: 8px 8px 0 0; padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────── DATA LOADING ──────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@st.cache_data(show_spinner="Loading datasets …")
def load_all():
    s = pd.read_csv(os.path.join(DATA_DIR, "SalesFINAL12312016.csv"))
    p = pd.read_csv(os.path.join(DATA_DIR, "PurchasesFINAL12312016.csv"))
    bi = pd.read_csv(os.path.join(DATA_DIR, "BegInvFINAL12312016.csv"))
    ei = pd.read_csv(os.path.join(DATA_DIR, "EndInvFINAL12312016.csv"))
    ip = pd.read_csv(os.path.join(DATA_DIR, "InvoicePurchases12312016.csv"))
    pp = pd.read_csv(os.path.join(DATA_DIR, "2017PurchasePricesDec.csv"))
    # clean
    for df in [s, p, bi, ei, ip, pp]:
        df.columns = df.columns.str.strip()
    s["SalesDate"] = pd.to_datetime(s["SalesDate"], errors="coerce")
    s["Revenue"] = s["SalesDollars"]
    s["YearMonth"] = s["SalesDate"].dt.to_period("M")
    s["Month"] = s["SalesDate"].dt.month
    s["DayOfWeek"] = s["SalesDate"].dt.dayofweek
    p["PODate"] = pd.to_datetime(p["PODate"], errors="coerce")
    p["ReceivingDate"] = pd.to_datetime(p["ReceivingDate"], errors="coerce")
    p["LeadTime"] = (p["ReceivingDate"] - p["PODate"]).dt.days
    p.loc[p["LeadTime"] < 0, "LeadTime"] = np.nan
    p.loc[p["LeadTime"] > 365, "LeadTime"] = np.nan
    p["YearMonth"] = p["PODate"].dt.to_period("M")
    p["UnitCost"] = np.where(p["Quantity"] > 0, p["Dollars"] / p["Quantity"], np.nan)
    return s, p, bi, ei, ip, pp

sales, purchases, beg_inv, end_inv, invoices, prices = load_all()

# ────────────────────────────── HELPERS ──────────────────────────────
def kpi(label, value, css=""):
    st.markdown(f'<div class="kpi-card {css}"><h3>{label}</h3><h1>{value}</h1></div>',
                unsafe_allow_html=True)

COLORS = ["#667eea","#764ba2","#2193b0","#11998e","#f7971e","#e74c3c",
          "#38ef7d","#6dd5ed","#ffd200","#c0392b","#8e2de2","#4a00e0"]

# ────────────────────────────── SIDEBAR ──────────────────────────────
st.sidebar.markdown("## 🍷 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📊 ABC Analysis",
    "📈 Demand Forecasting",
    "📦 EOQ Analysis",
    "🔄 Reorder Points",
    "🕐 Lead Time Analysis",
    "🏭 Supplier Performance",
    "📉 Inventory Turnover",
    "💡 Additional Insights",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Sales rows:** {len(sales):,}")
st.sidebar.markdown(f"**Purchase rows:** {len(purchases):,}")
st.sidebar.markdown(f"**Brands:** {sales['Brand'].nunique():,}")
st.sidebar.markdown(f"**Stores:** {int(sales['Store'].nunique())}")
st.sidebar.markdown(f"**Suppliers:** {purchases['VendorName'].nunique()}")

# ══════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🍷 Wine & Spirits Inventory Optimization Suite")
    st.markdown("*Comprehensive data-driven analysis for multi-location retail inventory management*")
    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi("Total Revenue", f"${sales['Revenue'].sum():,.0f}", "")
    with c2: kpi("Total Units Sold", f"{sales['SalesQuantity'].sum():,.0f}", "kpi-blue")
    with c3: kpi("Procurement Spend", f"${purchases['Dollars'].sum():,.0f}", "kpi-green")
    with c4: kpi("Avg Lead Time", f"{purchases['LeadTime'].mean():.1f} days", "kpi-orange")
    with c5: kpi("Active Brands", f"{sales['Brand'].nunique():,}", "kpi-purple")
    with c6: kpi("Stores", f"{int(sales['Store'].nunique())}", "kpi-teal")

    st.markdown("### 📊 Revenue by Month")
    monthly = sales.groupby(sales["SalesDate"].dt.to_period("M")).agg(
        Revenue=("Revenue","sum"), Qty=("SalesQuantity","sum")).reset_index()
    monthly["SalesDate"] = monthly["SalesDate"].astype(str)
    fig = px.area(monthly, x="SalesDate", y="Revenue", color_discrete_sequence=["#667eea"],
                  labels={"SalesDate":"Month","Revenue":"Revenue ($)"})
    fig.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏬 Top 10 Stores by Revenue")
        store_rev = sales.groupby("Store")["Revenue"].sum().nlargest(10).reset_index()
        store_rev["Store"] = store_rev["Store"].astype(int).astype(str)
        fig = px.bar(store_rev, x="Store", y="Revenue", color="Revenue",
                     color_continuous_scale="Viridis")
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏷️ Top 10 Brands by Revenue")
        brand_rev = sales.groupby("Brand")["Revenue"].sum().nlargest(10).reset_index()
        fig = px.bar(brand_rev, y="Brand", x="Revenue", orientation="h",
                     color="Revenue", color_continuous_scale="Purples")
        fig.update_layout(template="plotly_white", height=350, showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📅 Revenue by Day of Week")
        dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow = sales.groupby("DayOfWeek")["Revenue"].sum().reset_index()
        dow["Day"] = dow["DayOfWeek"].map(lambda x: dow_names[x])
        fig = px.bar(dow, x="Day", y="Revenue", color="Revenue",
                     color_continuous_scale=["#667eea","#e74c3c"])
        fig.update_layout(template="plotly_white", height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "Classification" in sales.columns:
            st.markdown("### 🏷️ Revenue by Classification")
            cls_rev = sales.groupby("Classification")["Revenue"].sum().reset_index()
            cls_rev["Classification"] = cls_rev["Classification"].astype(str)
            fig = px.pie(cls_rev, names="Classification", values="Revenue",
                         color_discrete_sequence=COLORS, hole=0.45)
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: ABC ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 ABC Analysis":
    st.markdown("# 📊 ABC Analysis (Pareto Classification)")
    st.markdown("Classifying inventory into **A** (high value, top 70%), **B** (moderate, 70-90%), **C** (low priority, 90-100%)")
    st.markdown("---")

    br = sales.groupby("Brand").agg(Revenue=("Revenue","sum"),Qty=("SalesQuantity","sum"),
         Txns=("Revenue","count")).sort_values("Revenue",ascending=False).reset_index()
    br["Revenue%"] = br["Revenue"]/br["Revenue"].sum()*100
    br["Cumulative%"] = br["Revenue%"].cumsum()
    br["Category"] = br["Cumulative%"].apply(lambda x: "A" if x<=70 else ("B" if x<=90 else "C"))

    cat_summary = br.groupby("Category").agg(Brands=("Brand","count"),Revenue=("Revenue","sum")).reset_index()

    c1,c2,c3,c4 = st.columns(4)
    a_row = cat_summary[cat_summary["Category"]=="A"]
    b_row = cat_summary[cat_summary["Category"]=="B"]
    c_row = cat_summary[cat_summary["Category"]=="C"]
    with c1: kpi("A-Category Brands", f"{int(a_row['Brands'].values[0]) if len(a_row) else 0}", "kpi-green")
    with c2: kpi("B-Category Brands", f"{int(b_row['Brands'].values[0]) if len(b_row) else 0}", "kpi-orange")
    with c3: kpi("C-Category Brands", f"{int(c_row['Brands'].values[0]) if len(c_row) else 0}", "kpi-red")
    with c4: kpi("Total Brands", f"{len(br):,}", "kpi-blue")

    col1, col2 = st.columns([2,1])
    with col1:
        top = br.head(30)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        color_map = {"A":"#2ecc71","B":"#f39c12","C":"#e74c3c"}
        fig.add_trace(go.Bar(x=top["Brand"].astype(str), y=top["Revenue"],
            marker_color=[color_map[c] for c in top["Category"]], name="Revenue", opacity=0.85))
        fig.add_trace(go.Scatter(x=top["Brand"].astype(str), y=top["Cumulative%"],
            mode="lines+markers", name="Cumulative %", line=dict(color="#2c3e50",width=3)), secondary_y=True)
        fig.add_hline(y=70, line_dash="dash", line_color="#2ecc71", annotation_text="A/B (70%)", secondary_y=True)
        fig.add_hline(y=90, line_dash="dash", line_color="#f39c12", annotation_text="B/C (90%)", secondary_y=True)
        fig.update_layout(title="Pareto Chart - Top 30 Brands", template="plotly_white",
                          height=450, xaxis_tickangle=-45, xaxis_tickfont_size=8)
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(cat_summary, names="Category", values="Revenue",
                     color="Category", color_discrete_map=color_map, hole=0.5,
                     title="Revenue Distribution")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Full ABC Table", expanded=False):
        cat_filter = st.multiselect("Filter Category", ["A","B","C"], default=["A","B","C"])
        st.dataframe(br[br["Category"].isin(cat_filter)][["Brand","Revenue","Qty","Revenue%","Cumulative%","Category"]].head(100),
                     use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.markdown("# 📈 Demand Forecasting (SARIMAX)")
    st.markdown("Weekly SARIMAX(1,1,1) time-series forecasts with 95% confidence intervals")
    st.markdown("---")

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    top_brands = sales.groupby("Brand")["SalesQuantity"].sum().nlargest(15).index.tolist()
    sel = st.selectbox("Select Brand to Forecast", top_brands)

    df_b = sales[sales["Brand"]==sel]
    weekly = df_b.groupby(pd.Grouper(key="SalesDate",freq="W"))["SalesQuantity"].sum().fillna(0)

    if len(weekly[weekly>0]) >= 4:
        try:
            model = SARIMAX(weekly, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
            fc = fit.get_forecast(steps=4)
            pred = fc.predicted_mean.clip(lower=0)
            ci = fc.conf_int().clip(lower=0)

            mae = np.abs(weekly - fit.fittedvalues).mean()
            rmse = np.sqrt(((weekly - fit.fittedvalues)**2).mean())

            c1,c2,c3,c4 = st.columns(4)
            with c1: kpi("MAE", f"{mae:.0f}", "kpi-blue")
            with c2: kpi("RMSE", f"{rmse:.0f}", "kpi-orange")
            with c3: kpi("AIC", f"{fit.aic:.0f}", "kpi-purple")
            with c4: kpi("Avg Forecast", f"{pred.mean():,.0f}/wk", "kpi-green")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly.values, mode="lines+markers",
                name="Historical", line=dict(color="#2c3e50", width=2)))
            fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines+markers",
                name="Forecast", line=dict(color="#e74c3c", width=3, dash="dash")))
            fig.add_trace(go.Scatter(x=ci.index.tolist()+ci.index.tolist()[::-1],
                y=ci.iloc[:,1].tolist()+ci.iloc[:,0].tolist()[::-1],
                fill="toself", fillcolor="rgba(231,76,60,0.15)", line=dict(width=0), name="95% CI"))
            fig.update_layout(title=f"Weekly Demand Forecast - Brand {sel}",
                template="plotly_white", height=450, xaxis_title="Week", yaxis_title="Quantity")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
    else:
        st.warning("Insufficient data for this brand (need 4+ weeks of non-zero sales).")

# ══════════════════════════════════════════════════════════════════════
#  PAGE: EOQ
# ══════════════════════════════════════════════════════════════════════
elif page == "📦 EOQ Analysis":
    st.markdown("# 📦 Economic Order Quantity (EOQ)")
    st.markdown("Wilson EOQ formula: **EOQ = √(2DS / H)** — Ordering=$150, Holding=20% of unit price")
    st.markdown("---")

    ad = sales.groupby("Brand")["SalesQuantity"].sum().reset_index()
    ad.columns = ["Brand","AnnualDemand"]
    ap = prices.groupby("Brand")["PurchasePrice"].mean().reset_index()
    ap.columns = ["Brand","AvgPrice"]
    eoq = ad.merge(ap, on="Brand", how="left")
    eoq["AvgPrice"].fillna(eoq["AvgPrice"].median(), inplace=True)
    eoq["HoldingCost"] = eoq["AvgPrice"]*0.20
    eoq["EOQ"] = np.sqrt(2*eoq["AnnualDemand"]*150/eoq["HoldingCost"]).round(0).astype(int)
    eoq["OrdersPerYear"] = np.ceil(eoq["AnnualDemand"]/eoq["EOQ"]).astype(int)
    eoq["TotalOrderCost"] = eoq["OrdersPerYear"]*150
    eoq["TotalHoldCost"] = (eoq["EOQ"]/2)*eoq["HoldingCost"]
    eoq["TotalCost"] = eoq["TotalOrderCost"]+eoq["TotalHoldCost"]
    eoq = eoq.sort_values("AnnualDemand", ascending=False).reset_index(drop=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Brands Analyzed", f"{len(eoq):,}", "kpi-blue")
    with c2: kpi("Avg EOQ", f"{eoq['EOQ'].mean():,.0f} units", "kpi-green")
    with c3: kpi("Total Inv Cost", f"${eoq['TotalCost'].sum():,.0f}", "kpi-orange")
    with c4: kpi("Avg Orders/Year", f"{eoq['OrdersPerYear'].mean():.0f}", "kpi-purple")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(eoq, x="AnnualDemand", y="EOQ", color="TotalCost", size="TotalCost",
            color_continuous_scale="YlOrRd", title="EOQ vs Annual Demand",
            labels={"AnnualDemand":"Annual Demand","EOQ":"EOQ (units)","TotalCost":"Total Cost ($)"})
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top = eoq.head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["Brand"].astype(str), x=top["TotalOrderCost"], name="Ordering Cost",
            orientation="h", marker_color="#e74c3c"))
        fig.add_trace(go.Bar(y=top["Brand"].astype(str), x=top["TotalHoldCost"], name="Holding Cost",
            orientation="h", marker_color="#3498db"))
        fig.update_layout(barmode="stack", template="plotly_white", height=400,
            title="Cost Breakdown - Top 15", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View EOQ Data Table"):
        st.dataframe(eoq.head(100), use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: REORDER POINTS
# ══════════════════════════════════════════════════════════════════════
elif page == "🔄 Reorder Points":
    st.markdown("# 🔄 Reorder Point Analysis")
    st.markdown("**ROP = (Avg Daily Demand × Avg Lead Time) + Safety Stock** at 95% service level (Z=1.65)")
    st.markdown("---")

    dd = sales.groupby(["Brand","SalesDate"])["SalesQuantity"].sum().reset_index()
    ds = dd.groupby("Brand")["SalesQuantity"].agg(["mean","std","sum"]).reset_index()
    ds.columns = ["Brand","AvgDaily","StdDaily","TotalDemand"]
    ds["StdDaily"].fillna(0, inplace=True)
    lt = purchases.groupby("Brand")["LeadTime"].agg(["mean","std"]).reset_index()
    lt.columns = ["Brand","AvgLT","StdLT"]
    rop = ds.merge(lt, on="Brand", how="left")
    glt = purchases["LeadTime"].mean(); gslt = purchases["LeadTime"].std()
    rop["AvgLT"].fillna(glt, inplace=True); rop["StdLT"].fillna(gslt, inplace=True)
    rop["SafetyStock"] = (1.65*np.sqrt(rop["AvgLT"]*rop["StdDaily"]**2 + rop["AvgDaily"]**2*rop["StdLT"]**2)).round(0).astype(int)
    rop["ROP"] = ((rop["AvgDaily"]*rop["AvgLT"])+rop["SafetyStock"]).round(0).astype(int)
    rop = rop.sort_values("TotalDemand", ascending=False).reset_index(drop=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Brands Analyzed", f"{len(rop):,}", "kpi-blue")
    with c2: kpi("Avg ROP", f"{rop['ROP'].mean():,.0f} units", "kpi-green")
    with c3: kpi("Avg Safety Stock", f"{rop['SafetyStock'].mean():,.0f} units", "kpi-orange")
    with c4: kpi("Avg Lead Time", f"{glt:.1f} days", "kpi-purple")

    top = rop.head(20)
    fig = go.Figure()
    demand_lt = top["AvgDaily"]*top["AvgLT"]
    fig.add_trace(go.Bar(y=top["Brand"].astype(str), x=demand_lt, name="Demand during LT",
        orientation="h", marker_color="#3498db"))
    fig.add_trace(go.Bar(y=top["Brand"].astype(str), x=top["SafetyStock"], name="Safety Stock",
        orientation="h", marker_color="#e74c3c"))
    fig.update_layout(barmode="stack", template="plotly_white", height=550,
        title="Reorder Point Components - Top 20 Brands", yaxis=dict(autorange="reversed"),
        xaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Reorder Point Data"):
        st.dataframe(rop.head(100), use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: LEAD TIME
# ══════════════════════════════════════════════════════════════════════
elif page == "🕐 Lead Time Analysis":
    st.markdown("# 🕐 Lead Time Analysis")
    st.markdown("Supply chain efficiency assessment: distribution, trends, and bottleneck identification")
    st.markdown("---")

    lt_data = purchases["LeadTime"].dropna()
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Mean Lead Time", f"{lt_data.mean():.1f} days", "kpi-blue")
    with c2: kpi("Median Lead Time", f"{lt_data.median():.1f} days", "kpi-green")
    with c3: kpi("Std Deviation", f"{lt_data.std():.1f} days", "kpi-orange")
    with c4: kpi("95th Percentile", f"{lt_data.quantile(0.95):.0f} days", "kpi-red")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(purchases.dropna(subset=["LeadTime"]), x="LeadTime", nbins=50,
            title="Lead Time Distribution", color_discrete_sequence=["#3498db"],
            labels={"LeadTime":"Lead Time (days)"})
        fig.add_vline(x=lt_data.mean(), line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Mean: {lt_data.mean():.1f}")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mlt = purchases.groupby("YearMonth")["LeadTime"].agg(["mean","median"]).reset_index()
        mlt["YearMonth"] = mlt["YearMonth"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mlt["YearMonth"], y=mlt["mean"], mode="lines+markers",
            name="Mean", line=dict(color="#2c3e50", width=2)))
        fig.add_trace(go.Scatter(x=mlt["YearMonth"], y=mlt["median"], mode="lines+markers",
            name="Median", line=dict(color="#e67e22", width=2, dash="dash")))
        fig.update_layout(title="Lead Time Trend Over Time", template="plotly_white", height=400,
            yaxis_title="Lead Time (days)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚨 Bottleneck Suppliers")
    vlt = purchases.groupby("VendorName")["LeadTime"].agg(["mean","std","count"]).reset_index()
    vlt.columns = ["Vendor","AvgLT","StdLT","Orders"]
    vlt = vlt[vlt["Orders"]>=5].sort_values("AvgLT", ascending=False)
    fig = px.bar(vlt.head(15), y="Vendor", x="AvgLT", orientation="h", error_x="StdLT",
        color="AvgLT", color_continuous_scale="OrRd", title="Slowest 15 Suppliers",
        labels={"AvgLT":"Avg Lead Time (days)"})
    fig.update_layout(template="plotly_white", height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: SUPPLIER PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
elif page == "🏭 Supplier Performance":
    st.markdown("# 🏭 Supplier Performance Dashboard")
    st.markdown("---")

    sp = purchases.groupby("VendorName").agg(Qty=("Quantity","sum"), Spend=("Dollars","sum"),
        Orders=("Quantity","count"), AvgLT=("LeadTime","mean"), StdLT=("LeadTime","std")).reset_index()
    sp["UnitCost"] = sp["Spend"]/sp["Qty"]
    sp = sp.sort_values("Spend", ascending=False).reset_index(drop=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Suppliers", f"{len(sp)}", "kpi-blue")
    with c2: kpi("Total Spend", f"${sp['Spend'].sum():,.0f}", "kpi-green")
    with c3: kpi("Total Units", f"{sp['Qty'].sum():,.0f}", "kpi-orange")
    with c4: kpi("Top Supplier", sp.iloc[0]["VendorName"][:20], "kpi-purple")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sp.head(15), y="VendorName", x="Spend", orientation="h",
            color="Spend", color_continuous_scale="Blues", title="Top 15 by Spend")
        fig.update_layout(template="plotly_white", height=450, yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(sp[sp["Orders"]>=5], x="AvgLT", y="Qty", size="Spend",
            color="UnitCost", color_continuous_scale="RdYlGn_r",
            title="Supplier Landscape (size=spend, color=unit cost)",
            labels={"AvgLT":"Avg Lead Time","Qty":"Total Qty","UnitCost":"Unit Cost ($)"}, hover_name="VendorName")
        fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Supplier Data"):
        st.dataframe(sp, use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: INVENTORY TURNOVER
# ══════════════════════════════════════════════════════════════════════
elif page == "📉 Inventory Turnover":
    st.markdown("# 📉 Inventory Turnover Analysis")
    st.markdown("**Turnover Ratio = COGS / Avg Inventory Value** | **DSI = 365 / Turnover**")
    st.markdown("---")

    cogs = sales["SalesDollars"].sum()
    bv = (beg_inv["onHand"]*beg_inv["Price"]).sum()
    ev = (end_inv["onHand"]*end_inv["Price"]).sum()
    avg_val = (bv+ev)/2
    ot = cogs/avg_val if avg_val>0 else 0
    dsi = 365/ot if ot>0 else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("COGS", f"${cogs:,.0f}", "kpi-blue")
    with c2: kpi("Avg Inventory Value", f"${avg_val:,.0f}", "kpi-green")
    with c3: kpi("Turnover Ratio", f"{ot:.2f}x", "kpi-orange")
    with c4: kpi("Days Sales Inventory", f"{dsi:.0f} days", "kpi-purple")

    ss = sales.groupby("Store")["SalesDollars"].sum().reset_index(); ss.columns=["Store","COGS"]
    sb = beg_inv.copy(); sb["IV"]=sb["onHand"]*sb["Price"]
    sbv = sb.groupby("Store")["IV"].sum().reset_index(); sbv.columns=["Store","BegV"]
    se = end_inv.copy(); se["IV"]=se["onHand"]*se["Price"]
    sev = se.groupby("Store")["IV"].sum().reset_index(); sev.columns=["Store","EndV"]
    sd = ss.merge(sbv, on="Store", how="outer").merge(sev, on="Store", how="outer").fillna(0)
    sd["AvgV"]=(sd["BegV"]+sd["EndV"])/2
    sd["Turnover"]=np.where(sd["AvgV"]>0, sd["COGS"]/sd["AvgV"], 0)
    sd["DSI"]=np.where(sd["Turnover"]>0, 365/sd["Turnover"], np.nan)
    sd["Store"]=sd["Store"].astype(int).astype(str)
    sd = sd.sort_values("Turnover", ascending=False)

    fig = px.bar(sd, x="Store", y="Turnover", color="Turnover",
        color_continuous_scale=["#e74c3c","#f39c12","#27ae60"], title="Turnover by Store")
    fig.add_hline(y=ot, line_dash="dash", line_color="#2c3e50",
                  annotation_text=f"Overall: {ot:.2f}x")
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: ADDITIONAL INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "💡 Additional Insights":
    st.markdown("# 💡 Additional Insights & Business Intelligence")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🏬 Store Performance", "📦 Dead Stock", "💰 Profit Margins", "📊 Data Explorer"])

    with tab1:
        sdf = sales.groupby("Store").agg(Revenue=("Revenue","sum"), Qty=("SalesQuantity","sum"),
            Txns=("Revenue","count"), Brands=("Brand","nunique")).sort_values("Revenue",ascending=False).reset_index()
        sdf["AvgTxn"] = sdf["Revenue"]/sdf["Txns"]
        sdf["Store"] = sdf["Store"].astype(int).astype(str)
        col1,col2 = st.columns(2)
        with col1:
            fig = px.bar(sdf, x="Store", y="Revenue", color="Revenue",
                color_continuous_scale="Viridis", title="Revenue by Store")
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(sdf, x="Qty", y="Revenue", size="Txns", color="AvgTxn",
                color_continuous_scale="RdYlGn", hover_name="Store",
                title="Store Landscape (size=transactions, color=avg txn value)")
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ib = end_inv.groupby("Brand").agg(Stock=("onHand","sum"), Price=("Price","mean")).reset_index()
        ib["Value"] = ib["Stock"]*ib["Price"]
        sb2 = sales.groupby("Brand")["SalesQuantity"].sum().reset_index(); sb2.columns=["Brand","Sold"]
        ds = ib.merge(sb2, on="Brand", how="left"); ds["Sold"].fillna(0, inplace=True)
        ds["Class"] = ds.apply(lambda r: "Dead Stock" if r["Sold"]==0 else
            ("Slow Moving" if r["Stock"]/r["Sold"]>2 else
            ("Normal" if r["Stock"]/r["Sold"]>0.5 else "Fast Moving")) if r["Sold"]>0 else "Dead Stock", axis=1)
        cs = ds.groupby("Class").agg(Brands=("Brand","count"), Value=("Value","sum")).reset_index()
        col1,col2 = st.columns(2)
        cmap = {"Dead Stock":"#e74c3c","Slow Moving":"#f39c12","Normal":"#3498db","Fast Moving":"#27ae60"}
        with col1:
            fig = px.pie(cs, names="Class", values="Brands", color="Class", color_discrete_map=cmap,
                hole=0.45, title="Inventory Movement Classification")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(cs.sort_values("Value",ascending=False), x="Class", y="Value", color="Class",
                color_discrete_map=cmap, title="Inventory Value by Movement Class")
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        dead_val = ds[ds["Class"]=="Dead Stock"]["Value"].sum()
        st.error(f"⚠️ Dead Stock Value: **${dead_val:,.2f}** across **{len(ds[ds['Class']=='Dead Stock']):,}** brands — capital at risk!")

    with tab3:
        avgs = sales.groupby("Brand")["SalesPrice"].mean().reset_index(); avgs.columns=["Brand","SellPrice"]
        avgb = prices.groupby("Brand")["PurchasePrice"].mean().reset_index(); avgb.columns=["Brand","BuyPrice"]
        mg = avgs.merge(avgb, on="Brand", how="inner")
        mg["MarginPct"] = (mg["SellPrice"]-mg["BuyPrice"])/mg["SellPrice"]*100
        brev = sales.groupby("Brand")["Revenue"].sum().reset_index()
        mg = mg.merge(brev, on="Brand", how="left").sort_values("Revenue",ascending=False)
        c1,c2,c3 = st.columns(3)
        with c1: kpi("Avg Margin", f"{mg['MarginPct'].mean():.1f}%", "kpi-green")
        with c2: kpi("High Margin (>50%)", f"{len(mg[mg['MarginPct']>50])}", "kpi-blue")
        with c3: kpi("Negative Margin", f"{len(mg[mg['MarginPct']<0])}", "kpi-red")
        col1,col2 = st.columns(2)
        with col1:
            top_m = mg.head(20)
            fig = px.bar(top_m, y="Brand", x="MarginPct", orientation="h",
                color="MarginPct", color_continuous_scale="RdYlGn",
                title="Margin % - Top 20 Brands by Revenue")
            fig.update_layout(template="plotly_white", height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(mg, x="MarginPct", nbins=40, title="Margin Distribution",
                color_discrete_sequence=["#667eea"])
            fig.add_vline(x=mg["MarginPct"].mean(), line_dash="dash", line_color="#e74c3c",
                annotation_text=f"Mean: {mg['MarginPct'].mean():.1f}%")
            fig.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 🔍 Explore Raw Data")
        dataset = st.selectbox("Select Dataset", ["Sales","Purchases","Beginning Inventory","Ending Inventory","Prices"])
        dmap = {"Sales":sales,"Purchases":purchases,"Beginning Inventory":beg_inv,
                "Ending Inventory":end_inv,"Prices":prices}
        chosen = dmap[dataset]
        st.write(f"**Shape:** {chosen.shape[0]:,} rows × {chosen.shape[1]} columns")
        st.dataframe(chosen.head(500), use_container_width=True, height=500)
