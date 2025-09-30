"""
AstraZeneca Supply Chain Analysis - Streamlit Dashboard
EDA Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="EDA - Supply Chain Analysis", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Check if data is loaded
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("⚠️ Please load data from the Home page first!")
    st.stop()

# Get data from session state
train_df = st.session_state.train_df
test_df = st.session_state.test_df

# Title
st.markdown('<div class="main-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)

# Sidebar for EDA options
with st.sidebar:
    st.markdown("### 🔧 EDA Options")
    
    eda_section = st.selectbox(
        "Select Analysis Section",
        [
            "Overview Statistics",
            "Data Quality",
            "Target Variables",
            "Temporal Analysis",
            "Product Analysis",
            "Supply Chain Network",
            "Cost Analysis",
            "Risk Analysis",
            "Correlation Analysis"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Visualization Style")
    plot_style = st.selectbox("Plot Type", ["Plotly (Interactive)", "Matplotlib (Static)"])

# Main content based on selection
if eda_section == "Overview Statistics":
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Records", f"{len(train_df):,}")
    with col2:
        st.metric("Test Records", f"{len(test_df):,}")
    with col3:
        st.metric("Training Features", train_df.shape[1])
    with col4:
        st.metric("Test Features", test_df.shape[1])
    
    st.markdown("---")
    
    # Data types
    st.markdown("### 📋 Data Types Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        dtype_counts = train_df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, 
                     title="Feature Data Types")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        cat_cols = train_df.select_dtypes(include=['object']).columns
        date_cols = train_df.select_dtypes(include=['datetime64']).columns
        
        st.markdown("#### Feature Breakdown")
        st.write(f"**Numeric Features:** {len(numeric_cols)}")
        st.write(f"**Categorical Features:** {len(cat_cols)}")
        st.write(f"**Datetime Features:** {len(date_cols)}")
        st.write(f"**Total Features:** {train_df.shape[1]}")
    
    # Basic statistics
    st.markdown("### 📈 Descriptive Statistics")
    st.dataframe(train_df.describe(), use_container_width=True)

elif eda_section == "Data Quality":
    st.markdown("## 🔍 Data Quality Analysis")
    
    # Missing values
    st.markdown("### ❌ Missing Values Analysis")
    
    missing = train_df.isnull().sum()
    missing_pct = (missing / len(train_df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Feature': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if len(missing_df) > 0:
            fig = px.bar(missing_df.head(20), x='Missing_Percentage', y='Feature',
                        orientation='h', title='Top 20 Features with Missing Values',
                        labels={'Missing_Percentage': 'Missing %'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with col2:
        st.markdown("#### Summary")
        st.metric("Features with Missing Values", len(missing_df))
        st.metric("Total Missing Values", f"{missing_df['Missing_Count'].sum():,}")
        
        if len(missing_df) > 0:
            st.metric("Highest Missing %", f"{missing_df['Missing_Percentage'].max():.2f}%")
    
    # Detailed table
    if len(missing_df) > 0:
        st.markdown("### 📋 Detailed Missing Values Table")
        st.dataframe(missing_df, use_container_width=True)
    
    # Duplicates
    st.markdown("### 🔄 Duplicate Records")
    col1, col2 = st.columns(2)
    
    with col1:
        train_dups = train_df.duplicated().sum()
        st.metric("Training Duplicates", train_dups)
    
    with col2:
        test_dups = test_df.duplicated().sum()
        st.metric("Test Duplicates", test_dups)
    
    # Data inconsistencies
    st.markdown("### ⚠️ Data Inconsistencies")
    
    issues = []
    
    if 'actual_delivery_days' in train_df.columns:
        neg_delivery = (train_df['actual_delivery_days'] < 0).sum()
        if neg_delivery > 0:
            issues.append(f"Negative actual delivery days: {neg_delivery:,}")
    
    if 'total_lead_time_days' in train_df.columns:
        neg_lead = (train_df['total_lead_time_days'] < 0).sum()
        if neg_lead > 0:
            issues.append(f"Negative total lead time: {neg_lead:,}")
    
    if issues:
        for issue in issues:
            st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ No major data inconsistencies detected!")

elif eda_section == "Target Variables":
    st.markdown("## 🎯 Target Variables Analysis")
    
    # Task A: Risk Classification
    if 'supply_chain_disruption_risk' in train_df.columns:
        st.markdown("### Task A: Supply Chain Disruption Risk")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            risk_counts = train_df['supply_chain_disruption_risk'].value_counts()
            fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                        title='Risk Level Distribution',
                        labels={'x': 'Risk Level', 'y': 'Count'},
                        color=risk_counts.index,
                        color_discrete_map={'Low_Risk': 'green', 'Medium_Risk': 'orange', 'High_Risk': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribution")
            for risk, count in risk_counts.items():
                pct = (count / len(train_df)) * 100
                st.metric(risk, f"{count:,} ({pct:.2f}%)")
    
    st.markdown("---")
    
    # Task B: Cost Prediction
    if 'realistic_total_cost' in train_df.columns:
        st.markdown("### Task B: Realistic Total Cost")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(train_df, x='realistic_total_cost', nbins=50,
                             title='Cost Distribution',
                             labels={'realistic_total_cost': 'Cost ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistics")
            st.metric("Mean", f"${train_df['realistic_total_cost'].mean():,.2f}")
            st.metric("Median", f"${train_df['realistic_total_cost'].median():,.2f}")
            st.metric("Std Dev", f"${train_df['realistic_total_cost'].std():,.2f}")
            st.metric("Min", f"${train_df['realistic_total_cost'].min():,.2f}")
            st.metric("Max", f"${train_df['realistic_total_cost'].max():,.2f}")
    
    st.markdown("---")
    
    # Task C: Delivery Delay
    if 'will_be_delayed' in train_df.columns:
        st.markdown("### Task C: Delivery Delay Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            delay_counts = train_df['will_be_delayed'].value_counts()
            labels = ['On-time', 'Delayed']
            fig = px.pie(values=delay_counts.values, names=labels,
                        title='Delivery Status Distribution',
                        color=labels,
                        color_discrete_map={'On-time': 'green', 'Delayed': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribution")
            on_time = delay_counts.get(0, 0)
            delayed = delay_counts.get(1, 0)
            st.metric("On-time (0)", f"{on_time:,} ({on_time/len(train_df)*100:.2f}%)")
            st.metric("Delayed (1)", f"{delayed:,} ({delayed/len(train_df)*100:.2f}%)")
    
    st.markdown("---")
    
    # Task D: Optimal Stock Days
    if 'optimal_stock_days' in train_df.columns:
        st.markdown("### Task D: Optimal Stock Days")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(train_df, x='optimal_stock_days', nbins=50,
                             title='Optimal Stock Days Distribution',
                             labels={'optimal_stock_days': 'Days'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistics")
            st.metric("Mean", f"{train_df['optimal_stock_days'].mean():.2f} days")
            st.metric("Median", f"{train_df['optimal_stock_days'].median():.2f} days")
            st.metric("Std Dev", f"{train_df['optimal_stock_days'].std():.2f} days")

elif eda_section == "Temporal Analysis":
    st.markdown("## 📅 Temporal Analysis")
    
    if 'order_date' in train_df.columns:
        # Date range
        st.markdown("### 📊 Date Range Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Start Date", train_df['order_date'].min().strftime('%Y-%m-%d'))
        with col2:
            st.metric("End Date", train_df['order_date'].max().strftime('%Y-%m-%d'))
        with col3:
            days = (train_df['order_date'].max() - train_df['order_date'].min()).days
            st.metric("Total Days", f"{days:,}")
        
        st.markdown("---")
        
        # Monthly trends
        st.markdown("### 📈 Monthly Order Trends")
        
        train_df['year_month'] = train_df['order_date'].dt.to_period('M').astype(str)
        monthly_orders = train_df.groupby('year_month').size().reset_index(name='count')
        
        fig = px.line(monthly_orders, x='year_month', y='count',
                     title='Orders Over Time',
                     labels={'year_month': 'Month', 'count': 'Number of Orders'})
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
        # Quarterly analysis
        st.markdown("### 📊 Quarterly Distribution")
        
        quarter_counts = train_df['quarter'].value_counts().sort_index()
        fig = px.bar(x=quarter_counts.index, y=quarter_counts.values,
                    title='Orders by Quarter',
                    labels={'x': 'Quarter', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week analysis
        st.markdown("### 📆 Day of Week Distribution")
        
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = train_df['day_of_week'].value_counts().reindex(dow_order, fill_value=0)
        
        fig = px.bar(x=dow_counts.index, y=dow_counts.values,
                    title='Orders by Day of Week',
                    labels={'x': 'Day', 'y': 'Count'},
                    color=dow_counts.values,
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)

elif eda_section == "Product Analysis":
    st.markdown("## 📦 Product Category Analysis")
    
    # Product categories
    st.markdown("### 🏷️ Product Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cat_counts = train_df['product_category'].value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                    title='Product Category Distribution',
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Categories")
        for cat, count in cat_counts.items():
            pct = (count / len(train_df)) * 100
            st.write(f"**{cat}:** {count:,} ({pct:.2f}%)")
    
    st.markdown("---")
    
    # Formulation types
    st.markdown("### 💊 Formulation Types")
    
    form_counts = train_df['formulation'].value_counts()
    fig = px.bar(x=form_counts.index, y=form_counts.values,
                title='Formulation Type Distribution',
                labels={'x': 'Formulation', 'y': 'Count'},
                color=form_counts.values,
                color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature requirements
    st.markdown("### 🌡️ Temperature Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp_counts = train_df['temperature_requirement'].value_counts()
        fig = px.bar(x=temp_counts.index, y=temp_counts.values,
                    title='Temperature Requirements',
                    labels={'x': 'Temperature', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'cold_chain_required' in train_df.columns:
            cold_chain = train_df['cold_chain_required'].value_counts()
            labels = ['Not Required', 'Required']
            fig = px.pie(values=[cold_chain.get(0.0, 0), cold_chain.get(1.0, 0)],
                        names=labels,
                        title='Cold Chain Requirements')
            st.plotly_chart(fig, use_container_width=True)

elif eda_section == "Supply Chain Network":
    st.markdown("## 🌐 Supply Chain Network Analysis")
    
    # Manufacturing sites
    st.markdown("### 🏭 Manufacturing Sites")
    
    mfg_counts = train_df['manufacturing_site'].value_counts()
    fig = px.bar(x=mfg_counts.values, y=mfg_counts.index,
                orientation='h',
                title='Manufacturing Sites Distribution',
                labels={'x': 'Count', 'y': 'Site'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution centers
    st.markdown("### 📦 Distribution Centers")
    
    dc_counts = train_df['distribution_center'].value_counts()
    fig = px.bar(x=dc_counts.values, y=dc_counts.index,
                orientation='h',
                title='Distribution Centers',
                labels={'x': 'Count', 'y': 'Distribution Center'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Market regions
    st.markdown("### 🌍 Market Regions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        region_counts = train_df['market_region'].value_counts()
        fig = px.pie(values=region_counts.values, names=region_counts.index,
                    title='Market Region Distribution',
                    hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Regional Breakdown")
        for region, count in region_counts.items():
            pct = (count / len(train_df)) * 100
            st.write(f"**{region}:** {count:,} ({pct:.2f}%)")
    
    # Shipping modes
    st.markdown("### 🚚 Shipping Modes")
    
    ship_counts = train_df['shipping_mode'].value_counts()
    fig = px.bar(x=ship_counts.index, y=ship_counts.values,
                title='Shipping Mode Distribution',
                labels={'x': 'Shipping Mode', 'y': 'Count'},
                color=ship_counts.values,
                color_continuous_scale='teal')
    st.plotly_chart(fig, use_container_width=True)

elif eda_section == "Cost Analysis":
    st.markdown("## 💰 Cost Analysis")
    
    cost_cols = ['api_cost_per_kg', 'excipient_cost_per_kg', 'packaging_cost_per_unit',
                 'manufacturing_cost_per_batch', 'total_cost_per_batch', 'realistic_total_cost']
    
    available_cost_cols = [col for col in cost_cols if col in train_df.columns]
    
    if available_cost_cols:
        # Select cost component
        selected_cost = st.selectbox("Select Cost Component", available_cost_cols)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(train_df, x=selected_cost, nbins=50,
                             title=f'{selected_cost.replace("_", " ").title()} Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistics")
            st.metric("Mean", f"${train_df[selected_cost].mean():,.2f}")
            st.metric("Median", f"${train_df[selected_cost].median():,.2f}")
            st.metric("Std Dev", f"${train_df[selected_cost].std():,.2f}")
            st.metric("Min", f"${train_df[selected_cost].min():,.2f}")
            st.metric("Max", f"${train_df[selected_cost].max():,.2f}")
        
        # Box plot
        st.markdown("### 📊 Cost Distribution by Category")
        
        if 'product_category' in train_df.columns:
            fig = px.box(train_df, x='product_category', y=selected_cost,
                        title=f'{selected_cost.replace("_", " ").title()} by Product Category',
                        labels={'product_category': 'Category', selected_cost: 'Cost ($)'})
            st.plotly_chart(fig, use_container_width=True)

elif eda_section == "Risk Analysis":
    st.markdown("## ⚠️ Risk Factor Analysis")
    
    # Supplier risk
    if 'supplier_risk_score' in train_df.columns:
        st.markdown("### 📊 Supplier Risk Score")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(train_df, x='supplier_risk_score', nbins=50,
                             title='Supplier Risk Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistics")
            st.metric("Mean", f"{train_df['supplier_risk_score'].mean():.2f}")
            st.metric("Median", f"{train_df['supplier_risk_score'].median():.2f}")
            st.metric("Min", f"{train_df['supplier_risk_score'].min():.2f}")
            st.metric("Max", f"{train_df['supplier_risk_score'].max():.2f}")
    
    # Stock out risk
    if 'stock_out_risk' in train_df.columns:
        st.markdown("### 📦 Stock Out Risk")
        
        fig = px.histogram(train_df, x='stock_out_risk', nbins=50,
                         title='Stock Out Risk Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Defect rate
    if 'defect_rate_ppm' in train_df.columns:
        st.markdown("### 🔍 Defect Rate (PPM)")
        
        fig = px.histogram(train_df, x='defect_rate_ppm', nbins=50,
                         title='Defect Rate Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality risk score
    if 'quality_risk_score' in train_df.columns:
        st.markdown("### ⭐ Quality Risk Score")
        
        fig = px.histogram(train_df, x='quality_risk_score', nbins=50,
                         title='Quality Risk Score Distribution')
        st.plotly_chart(fig, use_container_width=True)

elif eda_section == "Correlation Analysis":
    st.markdown("## 🔗 Correlation Analysis")
    
    # Select numeric columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to relevant columns
    key_vars = st.multiselect(
        "Select Variables for Correlation",
        numeric_cols,
        default=[col for col in ['supplier_risk_score', 'realistic_total_cost', 
                                 'optimal_stock_days', 'will_be_delayed'] 
                if col in numeric_cols][:4]
    )
    
    if len(key_vars) > 1:
        corr_matrix = train_df[key_vars].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix,
                       text_auto='.2f',
                       title='Correlation Heatmap',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation pairs
        st.markdown("### 🔝 Top Correlation Pairs")
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False, key=abs)
        st.dataframe(corr_df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**EDA Module** | AstraZeneca Supply Chain Analysis")
