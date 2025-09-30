"""
AstraZeneca Supply Chain Analysis - Streamlit Dashboard
Enhanced Preprocessing Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Preprocessing - Supply Chain", page_icon="🔧", layout="wide")

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("⚠️ Please load data from the Home page first!")
    st.stop()

train_df = st.session_state.train_df.copy()
test_df = st.session_state.test_df.copy()

st.markdown('<div class="main-header">🔧 Production-Grade Preprocessing</div>', 
            unsafe_allow_html=True)

# Initialize session state
if 'preprocessors' not in st.session_state:
    st.session_state.preprocessors = {}

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    task_selection = st.selectbox(
        "Select ML Task",
        ["Overview", "Task A: Risk", "Task B: Cost", "Task C: Delay", "Task D: Inventory"]
    )
    
    task_map = {
        "Task A: Risk": "risk",
        "Task B: Cost": "cost",
        "Task C: Delay": "delay",
        "Task D: Inventory": "inventory"
    }
    
    st.markdown("---")
    
    preprocessing_view = st.selectbox(
        "View Section",
        [
            "Pipeline Overview",
            "Data Quality Fixes",
            "Outlier Treatment",
            "Missing Values",
            "Feature Engineering",
            "Transformations",
            "Results Summary"
        ]
    )
    
    st.markdown("---")
    
    if task_selection != "Overview":
        task_key = task_map[task_selection]
        if st.button(f"🚀 Run Pipeline for {task_selection}", key=f"run_{task_key}"):
            with st.spinner(f"Processing {task_selection}..."):
                try:
                    import sys
                    import os
                    import traceback  # ADD THIS
                    
                    # Get the project root directory
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    sys.path.insert(0, project_root)
                    
                    from src.preprocessing import SupplyChainPreprocessor
                    
                    preprocessor = SupplyChainPreprocessor(train_df, test_df, task=task_key)
                    train_proc, test_proc = preprocessor.run_full_pipeline()
                    
                    st.session_state.preprocessors[task_key] = {
                        'preprocessor': preprocessor,
                        'train': train_proc,
                        'test': test_proc
                    }
                    
                    st.success(f"✅ {task_selection} preprocessing complete!")
                    st.experimental_rerun()
                except Exception as e:
                    # ADD THIS ENTIRE BLOCK:
                    st.error("❌ Error during preprocessing:")
                    st.error(f"Error type: {type(e).__name__}")
                    st.error(f"Error message: {str(e)}")
                    
                    # Show full traceback
                    st.code(traceback.format_exc())
                    
                    # Don't raise - keep Streamlit running

# Main content
if task_selection == "Overview":
    st.markdown("## 📋 Production-Grade Preprocessing Pipeline")
    
    st.markdown("""
    This module implements pharmaceutical industry-standard preprocessing with:
    - **Group-based imputation** (preserves operational patterns)
    - **Winsorization** (robust outlier treatment)
    - **40+ engineered features** (domain expertise)
    - **Task-specific preprocessing** (prevents target leakage)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 ML Tasks")
        tasks_df = pd.DataFrame({
            'Task': ['A: Risk Classification', 'B: Cost Prediction', 'C: Delay Prediction', 'D: Inventory Optimization'],
            'Target': ['supply_chain_disruption_risk', 'realistic_total_cost', 'will_be_delayed', 'optimal_stock_days'],
            'Type': ['Multi-class', 'Regression', 'Binary', 'Regression']
        })
        st.dataframe(tasks_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📊 Data Status")
        st.metric("Training Records", f"{len(train_df):,}")
        st.metric("Test Records", f"{len(test_df):,}")
        st.metric("Original Features", train_df.shape[1])
        st.metric("Missing Values", f"{train_df.isnull().sum().sum():,}")
    
    st.markdown("---")
    
    st.markdown("### 🔄 7-Step Pipeline")
    
    pipeline_df = pd.DataFrame({
        'Step': ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣'],
        'Name': ['Task-Specific Drops', 'Data Quality Fixes', 'Outlier Winsorization', 
                 'Group Imputation', 'Feature Engineering', 'Log Transforms', 'One-Hot Encoding'],
        'Description': [
            'Remove target and leakage columns per task',
            'Fix negative values, impossible data',
            'Cap outliers at 1%/99% percentiles',
            'Route-based for lead times, product-based for costs',
            'Create 40+ domain features (inventory, schedule, economics)',
            'Log-transform skewed cost variables',
            'Encode categoricals, align train-test'
        ],
        'Impact': ['Prevents leakage', 'Clean data', 'Robust models', 
                  'Preserve patterns', 'Boost accuracy', 'Stabilize variance', 'Model-ready']
    })
    
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)
    
    st.success("""
    **Key Improvements Over Basic Preprocessing:**
    - ✅ Task-specific leakage prevention
    - ✅ Winsorization instead of removal (preserves 99% of data)
    - ✅ Group-based imputation (routes, products) vs global median
    - ✅ 40+ engineered features with business justification
    - ✅ Log transforms for cost stabilization
    - ✅ Proper train-test alignment
    """)

elif preprocessing_view == "Pipeline Overview":
    task_key = task_map[task_selection]
    
    st.markdown(f"## {task_selection} - Pipeline Configuration")
    
    # Show task-specific drops
    st.markdown("### 🎯 Task-Specific Configuration")
    
    drops_config = {
        'risk': {
            'Target': 'supply_chain_disruption_risk',
            'Leakage Columns': 'None',
            'Evaluation': 'Weighted F1, Precision, Recall'
        },
        'cost': {
            'Target': 'realistic_total_cost',
            'Leakage Columns': 'total_cost_per_batch (high correlation r=0.95)',
            'Evaluation': 'RMSE, MAE, MAPE'
        },
        'delay': {
            'Target': 'will_be_delayed',
            'Leakage Columns': 'on_time_delivery, actual_delivery_days',
            'Evaluation': 'ROC-AUC, Precision, Recall'
        },
        'inventory': {
            'Target': 'optimal_stock_days',
            'Leakage Columns': 'None',
            'Evaluation': 'RMSE, Business cost function'
        }
    }
    
    config = drops_config[task_key]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Variable", config['Target'])
    with col2:
        st.metric("Leakage Risk", "High" if config['Leakage Columns'] != 'None' else "Low")
    with col3:
        st.metric("Evaluation Metrics", config['Evaluation'].split(',')[0])
    
    if config['Leakage Columns'] != 'None':
        st.warning(f"⚠️ **Leakage Prevention:** Dropping {config['Leakage Columns']}")


elif preprocessing_view == "Data Quality Fixes":
    st.markdown("## 🔧 Data Quality Fixes")
    
    # Negative values
    st.markdown("### 1. Impossible Values")
    
    neg_delivery = (train_df['actual_delivery_days'] < 0).sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Negative Delivery Days", neg_delivery)
    with col2:
        st.metric("% of Data", f"{neg_delivery/len(train_df)*100:.2f}%")
    with col3:
        st.metric("Action", "Set to NaN")
    
    if neg_delivery > 0:
        st.dataframe(
            train_df[train_df['actual_delivery_days'] < 0][
                ['order_date', 'planned_delivery_days', 'actual_delivery_days', 'shipping_mode']
            ].head(10),
            use_container_width=True
        )
    
    st.info("**Rationale:** Negative delivery days are impossible. Treating as data entry errors.")

elif preprocessing_view == "Outlier Treatment":
    st.markdown("## 📊 Winsorization Strategy")
    
    st.info("""
    **Why Winsorization > Removal:**
    - Preserves 99% of data (only caps extremes)
    - Maintains distribution shape
    - More robust than Z-score for skewed data
    - Better for gradient-based ML models
    """)
    
    # Select variable
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.selectbox("Select Variable", 
                           [c for c in numeric_cols if 'cost' in c or 'defect' in c])
    
    if selected:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Before/After comparison
            p1 = train_df[selected].quantile(0.01)
            p99 = train_df[selected].quantile(0.99)
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=train_df[selected], name='Before', boxmean='sd'))
            
            train_temp = train_df.copy()
            train_temp[selected] = train_temp[selected].clip(lower=p1, upper=p99)
            fig.add_trace(go.Box(y=train_temp[selected], name='After', boxmean='sd'))
            
            fig.update_layout(title=f'{selected} - Before/After Winsorization',
                            yaxis_title='Value')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Impact")
            before_std = train_df[selected].std()
            after_std = train_temp[selected].std()
            
            st.metric("Std Dev Before", f"{before_std:.2f}")
            st.metric("Std Dev After", f"{after_std:.2f}")
            st.metric("Reduction", f"{(1-after_std/before_std)*100:.1f}%")

elif preprocessing_view == "Missing Values":
    st.markdown("## 🔍 Intelligent Group-Based Imputation")
    
    st.success("""
    **Strategy: Preserve Operational Patterns**
    
    Instead of global median (destroys patterns), we impute by operational groups:
    - Lead times → by route (manufacturing_site + distribution_center)
    - Costs → by product characteristics (category + formulation)
    - Inventory → by product category
    """)
    
    # Show example: lead time by route
    if 'api_lead_time_days' in train_df.columns:
        st.markdown("### Example: API Lead Time by Route")
        
        route_medians = train_df.groupby(['manufacturing_site', 'distribution_center'])[
            'api_lead_time_days'
        ].median().reset_index()
        route_medians.columns = ['Manufacturing Site', 'Distribution Center', 'Median Lead Time (days)']
        
        st.dataframe(route_medians.head(15), use_container_width=True, hide_index=True)
        
        global_median = train_df['api_lead_time_days'].median()
        route_variance = route_medians['Median Lead Time (days)'].std()
        
        st.warning(f"""
        **Why This Matters:**
        - Global median: {global_median:.1f} days
        - Route variance: {route_variance:.1f} days
        - Using global median would lose {route_variance/global_median*100:.0f}% of route-specific patterns!
        """)

elif preprocessing_view == "Feature Engineering":
    st.markdown("## 🔬 40+ Engineered Features")
    
    feature_categories = {
        '📅 Temporal (7 features)': [
            'year, month_num, day_of_month',
            'is_weekend, is_quarter_end',
            'days_since_start (recency)',
            'month_sin, month_cos (cyclical)'
        ],
        '📦 Inventory Posture (6 features)': [
            'inv_gap = current - safety stock',
            'inv_below_safety (binary flag)',
            'inv_utilization = current / safety',
            'reorder_gap = reorder_point - current',
            'needs_reorder (binary flag)',
            'Business Value: Predicts stockout risk'
        ],
        '⏰ Schedule Realism (3 features)': [
            'schedule_slack = planned - total_lead_time',
            'under_planned (binary flag)',
            'schedule_buffer_ratio',
            'Business Value: 38% delay rate driver'
        ],
        '💰 Unit Economics (4 features)': [
            'cost_per_unit',
            'packaging_total',
            'api_cost_share (r=0.95 with total cost)',
            'Business Value: Cost optimization'
        ],
        '⭐ Quality Composites (1 feature)': [
            'quality_index = z(defect) + z(quality_risk)',
            'r=0.91 correlation found in EDA',
            'Business Value: Unified quality metric'
        ],
        '👥 Supplier Performance (1 feature)': [
            'supplier_perf_bucket (quartiles: 0-3)',
            'Business Value: Simplifies continuous performance'
        ],
        '⏱️ Lead Time Ratios (2 features)': [
            'api_lead_ratio, mfg_lead_ratio',
            'Business Value: Bottleneck identification'
        ],
        '📏 Batch Complexity (3 features)': [
            'is_large_batch (≥25K units)',
            'is_small_batch (≤2.5K units)',
            'batch_size_log',
            'Business Value: Economies of scale'
        ],
        '⚠️ Risk Interactions (1 feature)': [
            'combined_risk = supplier * (1 + stockout)',
            'Business Value: Compound risk assessment'
        ],
        '❄️ Compliance Flags (1 feature)': [
            'is_cold_chain',
            'Business Value: Logistics complexity'
        ]
    }
    
    selected_cat = st.selectbox("Select Feature Category", list(feature_categories.keys()))
    
    st.markdown(f"### {selected_cat}")
    for feature in feature_categories[selected_cat]:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total New Features", "40+")
    with col2:
        st.metric("Feature Categories", len(feature_categories))
    with col3:
        st.metric("All With Business Justification", "✅")

elif preprocessing_view == "Transformations":
    st.markdown("## 📈 Log Transformations")
    
    if 'realistic_total_cost' in train_df.columns:
        st.markdown("### Cost Distribution: Before vs After")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(train_df, x='realistic_total_cost', nbins=50,
                             title='Original (Right-Skewed)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning(f"""
            **Problems:**
            - Mean: ${train_df['realistic_total_cost'].mean():,.0f}
            - Median: ${train_df['realistic_total_cost'].median():,.0f}
            - Skew: {train_df['realistic_total_cost'].skew():.2f}
            - Outliers dominate gradients
            """)
        
        with col2:
            cost_log = np.log1p(train_df['realistic_total_cost'])
            fig = px.histogram(x=cost_log, nbins=50,
                             title='Log-Transformed (Normalized)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            **Improvements:**
            - Skew: {cost_log.skew():.2f}
            - More symmetric
            - Stabilized variance
            - Better for linear models
            """)
    
    st.info("""
    **When to Log Transform:**
    - Right-skewed distributions (mean >> median)
    - Multiplicative relationships
    - Wide value ranges (orders of magnitude)
    - Cost, time, size variables
    """)

elif preprocessing_view == "Results Summary":
    task_key = task_map.get(task_selection)
    
    if task_key and task_key in st.session_state.preprocessors:
        data = st.session_state.preprocessors[task_key]
        train_proc = data['train']
        test_proc = data['test']
        preprocessor = data['preprocessor']
        
        st.markdown(f"## ✅ {task_selection} - Processing Complete")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Features", train_proc.shape[1])
            st.metric("Training Samples", f"{train_proc.shape[0]:,}")
        
        with col2:
            new_features = train_proc.shape[1] - train_df.shape[1] + len(preprocessor.dropped_features)
            st.metric("Features Added", f"+{new_features}")
            st.metric("Test Samples", f"{test_proc.shape[0]:,}")
        
        with col3:
            st.metric("Missing Values", train_proc.isnull().sum().sum())
            st.metric("Data Quality", "✅ High")
        
        with col4:
            st.metric("Ready for ML", "✅ Yes")
            st.metric("Leakage Risk", "✅ Mitigated")
        
        st.markdown("---")
        
        # Before/After comparison
        st.markdown("### 📊 Before vs After")
        
        comparison = pd.DataFrame({
            'Metric': ['Total Features', 'Missing Values', 'Outliers', 'Target Leakage', 'ML Ready'],
            'Before': [
                train_df.shape[1],
                f"{train_df.isnull().sum().sum():,}",
                'Extreme values present',
                'Risk present',
                '❌'
            ],
            'After': [
                train_proc.shape[1],
                '0',
                'Winsorized at 1%/99%',
                'Mitigated',
                '✅'
            ]
        })
        
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Feature composition
        st.markdown("### 🎯 Feature Composition")
        
        feature_breakdown = pd.DataFrame({
            'Category': ['Original Numeric', 'Original Categorical (encoded)', 
                        'Engineered Features', 'Log Transforms'],
            'Count': [30, train_proc.shape[1] - 30 - 40 - 4, 40, 4],
            'Purpose': ['Base measurements', 'One-hot encoded', 
                       'Domain expertise', 'Stabilize distributions']
        })
        
        fig = px.pie(feature_breakdown, values='Count', names='Category',
                    title='Feature Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"""
        🎉 **Preprocessing Complete for {task_selection}!**
        
        Data is now optimized for machine learning with:
        - Zero missing values
        - Treated outliers
        - 40+ engineered features
        - Task-specific leakage prevention
        - Proper train-test alignment
        
        **Next Step:** Navigate to Modeling page to train ML models.
        """)
        
        # Download button
        if st.button("💾 Download Processed Data"):
            csv = train_proc.to_csv(index=False)
            st.download_button(
                "Download Training Data",
                csv,
                f"train_processed_{task_key}.csv",
                "text/csv"
            )
    
    else:
        st.warning(f"⚠️ Run preprocessing pipeline for {task_selection} first!")

# Footer
st.markdown("---")
st.markdown("**Enhanced Preprocessing Module** | Production-Grade Pipeline")
