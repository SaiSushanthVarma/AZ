"""
AstraZeneca Supply Chain Analysis - Streamlit Dashboard
Main Application
Author: [Your Name]
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="AstraZeneca Supply Chain Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #f0f2f6 0%, #e8eaf0 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None

# Load data function
@st.cache_data
def load_data():
    """Load data from parquet files"""
    try:
        train_df = pd.read_parquet('data/pharma_supply_chain_train.parquet')
        test_df = pd.read_parquet('data/pharma_supply_chain_test.parquet')
        return train_df, test_df
    except:
        return None, None

# Main page
def main():
    # Header
    st.markdown('<div class="main-header">💊 AstraZeneca Supply Chain Analysis</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        #st.image("https://via.placeholder.com/300x100/1f77b4/FFFFFF?text=AstraZeneca", 
        #        use_container_width=True)
        st.markdown("---")
        st.markdown("### 📋 Navigation")
        st.markdown("""
        This dashboard provides comprehensive analysis of pharmaceutical supply chain operations.
        
        **Sections:**
        - 🔍 **EDA**: Exploratory Data Analysis
        - 🔧 **Preprocessing**: Data cleaning & feature engineering
        - 🤖 **Modeling**: ML model development
        - 💡 **Business Insights**: Actionable recommendations
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        
        if st.button("🔄 Load Data", key="load_data"):
            with st.spinner("Loading data..."):
                train_df, test_df = load_data()
                if train_df is not None:
                    st.session_state.train_df = train_df
                    st.session_state.test_df = test_df
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                else:
                    st.error("❌ Failed to load data")
        
        if st.session_state.data_loaded:
            st.metric("Training Records", f"{len(st.session_state.train_df):,}")
            st.metric("Test Records", f"{len(st.session_state.test_df):,}")
            st.metric("Total Features", st.session_state.train_df.shape[1])
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data' in the sidebar to begin analysis")
        
        # Welcome section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3>🎯 Objective</h3>
            <p>Optimize pharmaceutical supply chain operations through predictive analytics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3>📦 Tasks</h3>
            <p>Risk Classification, Cost Prediction, Delay Forecasting, Inventory Optimization</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h3>🏆 Impact</h3>
            <p>Reduce costs, minimize stockouts, improve delivery performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Project overview
        st.markdown("## 📖 Project Overview")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔧 Preprocessing", "🤖 Modeling", "💡 Insights"])
        
        with tab1:
            st.markdown("""
            ### Exploratory Data Analysis
            - Data quality assessment
            - Missing value analysis
            - Temporal trends and seasonality
            - Product category distribution
            - Supply chain network analysis
            - Cost and risk factor analysis
            - Correlation studies
            """)
        
        with tab2:
            st.markdown("""
            ### Data Preprocessing
            - Missing value imputation
            - Outlier detection and treatment
            - Feature engineering
            - Encoding categorical variables
            - Scaling numerical features
            - Train-validation split
            """)
        
        with tab3:
            st.markdown("""
            ### Machine Learning Models
            
            **Task A: Supply Chain Disruption Risk**
            - Multi-class classification (Low/Medium/High Risk)
            - Models: Random Forest, XGBoost, LightGBM
            - Evaluation: Weighted F1-score, Precision, Recall
            
            **Task B: Total Cost Prediction**
            - Regression task
            - Models: Gradient Boosting, Random Forest
            - Evaluation: RMSE, MAE, MAPE
            
            **Task C: Delivery Delay Prediction**
            - Binary classification
            - Models: Logistic Regression, XGBoost
            - Evaluation: ROC-AUC, Precision, Recall
            
            **Task D: Inventory Optimization**
            - Regression task
            - Models: XGBoost, LightGBM
            - Evaluation: RMSE, Business cost function
            """)
        
        with tab4:
            st.markdown("""
            ### Business Insights & Recommendations
            - Key drivers of supply chain disruption
            - Cost optimization opportunities
            - Delivery performance improvement strategies
            - Inventory management recommendations
            - Risk mitigation approaches
            - Implementation roadmap
            - Expected ROI quantification
            """)
    
    else:
        # Data loaded - show summary
        st.success("✅ Data loaded successfully! Navigate to EDA page to explore the data.")
        
        # Quick stats
        st.markdown("## 📊 Quick Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders (Train)", f"{len(st.session_state.train_df):,}")
        
        with col2:
            product_categories = st.session_state.train_df['product_category'].nunique()
            st.metric("Product Categories", product_categories)
        
        with col3:
            mfg_sites = st.session_state.train_df['manufacturing_site'].nunique()
            st.metric("Manufacturing Sites", mfg_sites)
        
        with col4:
            if 'realistic_total_cost' in st.session_state.train_df.columns:
                avg_cost = st.session_state.train_df['realistic_total_cost'].mean()
                st.metric("Avg Total Cost", f"${avg_cost:,.0f}")
        
        st.markdown("---")
        
        # Data preview
        st.markdown("## 🔍 Data Preview")
        
        preview_option = st.radio("Select dataset:", ["Training Data", "Test Data"], horizontal=True)
        
        if preview_option == "Training Data":
            st.dataframe(st.session_state.train_df.head(20), use_container_width=True)
        else:
            st.dataframe(st.session_state.test_df.head(20), use_container_width=True)
        
        # Data info
        st.markdown("## 📋 Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Data")
            st.write(f"**Shape:** {st.session_state.train_df.shape}")
            st.write(f"**Columns:** {st.session_state.train_df.shape[1]}")
            st.write(f"**Rows:** {st.session_state.train_df.shape[0]:,}")
            
            missing_train = st.session_state.train_df.isnull().sum().sum()
            st.write(f"**Missing Values:** {missing_train:,}")
        
        with col2:
            st.markdown("### Test Data")
            st.write(f"**Shape:** {st.session_state.test_df.shape}")
            st.write(f"**Columns:** {st.session_state.test_df.shape[1]}")
            st.write(f"**Rows:** {st.session_state.test_df.shape[0]:,}")
            
            missing_test = st.session_state.test_df.isnull().sum().sum()
            st.write(f"**Missing Values:** {missing_test:,}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>AstraZeneca Supply Chain Analysis Dashboard | Data Science Assignment 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
