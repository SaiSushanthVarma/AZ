# Pharmaceutical Supply Chain ML Analysis

Machine learning models for optimizing pharmaceutical supply chain operations across risk classification, cost prediction, delay forecasting, and inventory management.

---

## Project Overview

This project analyzes 60,000 pharmaceutical shipment orders using four machine learning models to identify optimization opportunities and improve supply chain efficiency.

**Key Results:**
- Identified 486 high-risk shipments requiring intervention
- Predicted 1,395 delayed shipments for proactive customer communication
- Found 29,069 overstocked products for $363M working capital release
- Analyzed $7.1B in total spend for cost optimization

---

## Project Structure

```
AZ/
├── data/                           # Data storage (not included)
├── outputs/                        # Model outputs and results
│   ├── figures/                    # Visualizations and charts
│   ├── models/                     # Trained model files
│   └── predictions/                # CSV prediction files
│       ├── prediction_task_a_risk.csv
│       ├── prediction_task_b_cost.csv
│       ├── prediction_task_c_delay.csv
│       └── prediction_task_d_inventory.csv
├── src/                            # Source code
│   ├── 03_ml_forecasting/          # ML model training scripts
│   │   ├── task_a_risk.py          # Risk classification model
│   │   ├── task_b_cost.py          # Cost prediction model
│   │   ├── task_c_delay.py         # Delay prediction model
│   │   └── task_d_inventory.py    # Inventory optimization model
│   ├── 01_eda.py                   # Exploratory data analysis
│   ├── load_data.py                # Data loading utilities
│   └── preprocessing.py            # Data preprocessing functions
├── streamlit_app/                  # Web application (optional)
│   ├── pages/
│   │   ├── 1_EDA.py               # EDA dashboard
│   │   └── 2_Preprocessing.py     # Preprocessing interface
│   └── app.py                      # Main Streamlit app
├── full_correlation_matrix.csv     # Feature correlation analysis
├── requirements.txt                # Python dependencies
├── readme.md                       # This file
└── Take-home-assignment.pdf        # Project assignment document
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone or download the repository**

2. **Create virtual environment** (recommended)
```bash
python -m venv az_env
source az_env/bin/activate  # On Windows: az_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Running ML Models

Each task can be run independently:

```bash
# Task A: Risk Classification
python src/03_ml_forecasting/task_a_risk.py

# Task B: Cost Prediction
python src/03_ml_forecasting/task_b_cost.py

# Task C: Delay Prediction
python src/03_ml_forecasting/task_c_delay.py

# Task D: Inventory Optimization
python src/03_ml_forecasting/task_d_inventory.py
```

### Running EDA

```bash
python src/01_eda.py
```

### Running Streamlit Dashboard (Optional)

```bash
streamlit run streamlit_app/app.py
```

---

## Models and Performance

### Task A: Supply Chain Risk Classification
- **Model:** LightGBM Classifier
- **Performance:** F1-Score = 0.987, Cross-Validation F1 = 0.993
- **Output:** Risk levels (Low_Risk, Medium_Risk, High_Risk)
- **Business Impact:** Identifies 486 high-risk shipments for proactive intervention

### Task B: Total Cost Prediction
- **Model:** LightGBM Regressor
- **Performance:** MAPE = 25.6%, MAE = $25,674, RMSE = $33,150
- **Output:** Predicted cost per batch
- **Business Impact:** Improves budget accuracy and enables supplier negotiations

### Task C: Delivery Delay Prediction
- **Model:** XGBoost Classifier
- **Performance:** ROC-AUC = 0.626, Cross-Validation AUC = 0.588
- **Output:** Binary prediction (0 = On-time, 1 = Delayed)
- **Business Impact:** Enables proactive customer communication for 1,395 at-risk shipments

### Task D: Inventory Optimization
- **Model:** XGBoost Regressor
- **Performance:** RMSE = 13.7 days, MAE = 9.9 days
- **Output:** Optimal stock days per product
- **Business Impact:** Identifies $363M working capital optimization opportunity

---

## Output Files

All prediction files are located in `outputs/predictions/`:

**Format:** CSV with columns `order_id` and `prediction`

- `prediction_task_a_risk.csv` - Risk classifications
- `prediction_task_b_cost.csv` - Cost predictions
- `prediction_task_c_delay.csv` - Delay predictions (0/1)
- `prediction_task_d_inventory.csv` - Optimal stock days

---

## Business Recommendations

### Priority 1: Risk Management ($35K investment)
- Monitor 486 high-risk shipments daily
- Deploy enhanced tracking and dual sourcing
- **Expected savings:** $464K/year

### Priority 2: Customer Communication ($25K investment)
- Proactive alerts for 1,395 delayed shipments
- Offer alternatives and regional buffers
- **Expected savings:** $420K/year

### Priority 3: Inventory Optimization ($75K investment)
- Reduce inventory on 29,069 overstocked products
- Implement dynamic reorder points and FEFO
- **Expected savings:** $4M/year + $363M working capital

**Total Investment:** $135K  
**Total Annual Savings:** $4.9M  
**ROI:** 36x  
**Payback Period:** 10 days

---

## Key Features

- **Data Preprocessing:** Automated feature engineering and data cleaning
- **Multiple Models:** Ensemble approach with Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning:** GridSearchCV for optimal model parameters
- **Cross-Validation:** 5-fold CV for robust performance estimation
- **Business Metrics:** Custom evaluation metrics aligned with business objectives

---



## Results Summary

**Predictions Generated:** 60,000 orders analyzed

**Key Findings:**
- 0.8% high-risk shipments (486)
- 2.3% delayed shipments (1,395)
- 48.5% overstocked products (29,069)
- $7.1B total predicted spend

**Business Impact:**
- $4.9M annual cost savings identified
- $363M working capital release opportunity
- 36x ROI on $135K investment
- Improves on-time delivery from 97.7% to 98.8%

