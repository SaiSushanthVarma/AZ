import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


class SupplyChainPreprocessor:

    def __init__(self, train_df, test_df, task="all"):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.task = task
        self.train_processed = None
        self.test_processed = None
        self.dropped_features = []
        self.engineered_features = []
        self.train_stats = {}
        self.scaler = None

        # Ensure datetime
        for df in [self.train_df, self.test_df]:
            if "order_date" in df.columns:
                df["order_date"] = pd.to_datetime(df["order_date"])
            if "order_timestamp" in df.columns:
                df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])

# -------------------------------------------------------------------------
    # STEP 1. Drop leakage columns

    def drop_task_specific_columns(self):
        
        if "order_id" in self.test_df.columns:
            test_ids = pd.DataFrame({"order_id": self.test_df["order_id"]})
            test_ids.to_csv(f"data/test_order_ids_{self.task}.csv", index=False)
            print(f"Saved test order_ids to: data/test_order_ids_{self.task}.csv")

        always_drop = ["order_id", "risk_description", "batch_id", "supply_chain_complexity"]
        
        # drop specific columns for specific models of ML tasks
        task_drops = {
            "risk": ["supply_chain_disruption_risk"],
            "cost": ["realistic_total_cost", "total_cost_per_batch"],
            "delay": ["will_be_delayed", "on_time_delivery", "actual_delivery_days"],
            "inventory": ["optimal_stock_days"],
        }

        drops = always_drop.copy()
        if self.task != "all":
            drops.extend(task_drops.get(self.task, []))

        for col in drops:
            for df in [self.train_df, self.test_df]:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            self.dropped_features.append(col)

        print(f"Dropped {len(drops)} columns")
        return self

    
    def fix_data_quality(self):
        
        fixes = 0
        for df in [self.train_df, self.test_df]:
            if "actual_delivery_days" in df.columns:
                neg_mask = df["actual_delivery_days"] < 0
                fixes += neg_mask.sum()
                df.loc[neg_mask, "actual_delivery_days"] = np.nan

        print(f"Fixed {fixes} impossible values")
        return self

#capping top 1 % and bottom 1% of key cost and quality metrics
    def winsorize_outliers(self):

        outlier_cols = [
            "api_cost_per_kg", "excipient_cost_per_kg", "packaging_cost_per_unit",
            "manufacturing_cost_per_batch", "defect_rate_ppm", "batch_release_time_hours"
        ]

        for col in outlier_cols:
            if col in self.train_df.columns:
                p1 = self.train_df[col].quantile(0.01)
                p99 = self.train_df[col].quantile(0.99)
                self.train_stats[f"{col}_p01"] = p1
                self.train_stats[f"{col}_p99"] = p99
                for df in [self.train_df, self.test_df]:
                    df[col] = df[col].clip(lower=p1, upper=p99)


        return self

    def impute_missing_values(self):
        # Lead times
        lead_cols = [
            "api_lead_time_days", "excipient_lead_time_days", "packaging_lead_time_days",
            "manufacturing_lead_time_days", "total_lead_time_days", "planned_delivery_days"
        ]
        for col in lead_cols:
            if col in self.train_df.columns:
                group_medians = self.train_df.groupby(
                    ["manufacturing_site", "distribution_center"]
                )[col].median()
                global_median = self.train_df[col].median()
                for df in [self.train_df, self.test_df]:
                    for idx, row in df.iterrows():
                        if pd.isna(df.loc[idx, col]):
                            key = (row["manufacturing_site"], row["distribution_center"])
                            df.loc[idx, col] = group_medians.get(key, global_median)

        # Costs
        cost_cols = ["api_cost_per_kg", "excipient_cost_per_kg",
                     "packaging_cost_per_unit", "manufacturing_cost_per_batch"]
        for col in cost_cols:
            if col in self.train_df.columns:
                group_medians = self.train_df.groupby(
                    ["product_category", "formulation"]
                )[col].median()
                global_median = self.train_df[col].median()
                for df in [self.train_df, self.test_df]:
                    for idx, row in df.iterrows():
                        if pd.isna(df.loc[idx, col]):
                            key = (row["product_category"], row["formulation"])
                            df.loc[idx, col] = group_medians.get(key, global_median)

        # Inventory
        inv_cols = ["current_stock_days", "safety_stock_days", "reorder_point_days"]
        for col in inv_cols:
            if col in self.train_df.columns:
                group_medians = self.train_df.groupby("product_category")[col].median()
                global_median = self.train_df[col].median()
                for df in [self.train_df, self.test_df]:
                    df[col] = df.groupby("product_category")[col].transform(
                        lambda x: x.fillna(group_medians.get(x.name, global_median))
                    )
                    df[col].fillna(global_median, inplace=True)

        # Remaining numerics
        for col in self.train_df.select_dtypes(include=[np.number]).columns:
            if self.train_df[col].isnull().any():
                med = self.train_df[col].median()
                for df in [self.train_df, self.test_df]:
                    df[col].fillna(med, inplace=True)

        print("Imputed all missing values")
        return self
    # -------------------------------------------------------------------------
    # adding additional features that mightbe useful for supply chain 
    def engineer_features(self):
        print("\n[STEP 5] FEATURE ENGINEERING")
        print("-" * 80)

        for df in [self.train_df, self.test_df]:
            # Temporallater used in ML models
            if "order_date" in df.columns:
                df["month_num"] = df["order_date"].dt.month
                df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
                df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
                df["is_quarter_end"] = df["month_num"].isin([3, 6, 9, 12]).astype(int)


            if all(c in df.columns for c in ["current_stock_days", "safety_stock_days"]):
                df["inv_below_safety"] = (df["current_stock_days"] < df["safety_stock_days"]).astype(int)
            if all(c in df.columns for c in ["reorder_point_days", "current_stock_days"]):
                df["needs_reorder"] = (df["reorder_point_days"] > df["current_stock_days"]).astype(int)
            if all(c in df.columns for c in ["average_daily_unit_sales", "total_lead_time_days", "safety_stock_days"]):
                df["reorder_point_calc"] = (
                    df["average_daily_unit_sales"] * df["total_lead_time_days"]
                ) + df["safety_stock_days"]

           
            if all(c in df.columns for c in ["planned_delivery_days", "total_lead_time_days"]):
                df["schedule_slack"] = df["planned_delivery_days"] - df["total_lead_time_days"]
                df["under_planned"] = (df["schedule_slack"] < 0).astype(int)


            if all(c in df.columns for c in ["batch_size_units", "manufacturing_cost_per_batch"]):
                df["cost_per_unit"] = df["manufacturing_cost_per_batch"] / (df["batch_size_units"] + 1)
            if all(c in df.columns for c in ["api_cost_per_kg", "manufacturing_cost_per_batch"]):
                df["api_cost_share"] = df["api_cost_per_kg"] / (df["manufacturing_cost_per_batch"] + 1)
                df["api_cost_share"] = df["api_cost_share"].clip(0, 1)


            if "defect_rate_ppm" in df.columns and "quality_risk_score" in df.columns:
                defect_z = (df["defect_rate_ppm"] - self.train_df["defect_rate_ppm"].mean()) / (
                    self.train_df["defect_rate_ppm"].std() + 1e-8
                )
                quality_z = (df["quality_risk_score"] - self.train_df["quality_risk_score"].mean()) / (
                    self.train_df["quality_risk_score"].std() + 1e-8
                )
                df["quality_index"] = defect_z + quality_z
            if "supplier_historical_performance" in df.columns:
                q25 = self.train_df["supplier_historical_performance"].quantile(0.25)
                q50 = self.train_df["supplier_historical_performance"].quantile(0.50)
                q75 = self.train_df["supplier_historical_performance"].quantile(0.75)
                df["supplier_perf_bucket"] = pd.cut(
                    df["supplier_historical_performance"],
                    bins=[-np.inf, q25, q50, q75, np.inf],
                    labels=[0, 1, 2, 3]
                ).astype(int)

            if "cold_chain_required" in df.columns:
                df["is_cold_chain"] = df["cold_chain_required"].astype(int)

        print("Created essential engineered features")
        return self

    
    def normalize_numeric(self):
        # Log transform skewed costs
        cost_cols = ['api_cost_per_kg', 'excipient_cost_per_kg', 
                     'packaging_cost_per_unit', 'manufacturing_cost_per_batch']

        for col in cost_cols:
            if col in self.train_df.columns:
                new_col = f'{col}_log'
                for df in [self.train_df, self.test_df]:
                    df[new_col] = np.log1p(df[col])
                self.engineered_features.append(new_col)

        # Select numeric columns (only those actually present)
        num_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()

        # Fit scaler on train, transform both
        self.scaler = StandardScaler()
        self.train_df[num_cols] = self.scaler.fit_transform(self.train_df[num_cols])
        self.test_df[num_cols] = self.scaler.transform(self.test_df[num_cols].reindex(columns=num_cols, fill_value=0))

        print(f"Scaled {len(num_cols)} numeric columns")
        return self

    def encode_categoricals(self):
        
        cat_cols = [
            "product_category", "formulation", "temperature_requirement",
            "manufacturing_site", "distribution_center", "api_supplier",
            "excipient_supplier", "packaging_supplier", "shipping_mode",
            "geopolitical_risk", "market_region", "therapeutic_area_priority",
            "day_of_week", "quarter", "month"
        ]
        cat_cols = [c for c in cat_cols if c in self.train_df.columns]

        self.train_df = pd.get_dummies(self.train_df, columns=cat_cols, drop_first=True, dtype=int)
        self.test_df = pd.get_dummies(self.test_df, columns=cat_cols, drop_first=True, dtype=int)

        for col in set(self.train_df.columns) - set(self.test_df.columns):
            self.test_df[col] = 0
        for col in set(self.test_df.columns) - set(self.train_df.columns):
            self.train_df[col] = 0
        self.test_df = self.test_df[self.train_df.columns]

        print(f"Encoded {len(cat_cols)} categorical variables")
        return self

    def run_all_above(self):
        
        self.drop_task_specific_columns()
        self.fix_data_quality()
        self.winsorize_outliers()
        self.impute_missing_values()
        self.engineer_features()
        self.normalize_numeric()
        self.encode_categoricals()

        self.train_processed = self.train_df.copy()
        self.test_processed = self.test_df.copy()

        print(f"Training: {self.train_processed.shape}")
        print(f"Test: {self.test_processed.shape}")
        print(f"Missing values: {self.train_processed.isnull().sum().sum()}")
        return self.train_processed, self.test_processed

    def save_processed_data(self, output_dir='data/'):
        """Save processed datasets"""
        if self.train_processed is not None:
            train_path = f'{output_dir}train_processed_{self.task}.csv'
            test_path = f'{output_dir}test_processed_{self.task}.csv'
            
            self.train_processed.to_csv(train_path, index=False)
            self.test_processed.to_csv(test_path, index=False)
            
            print(f"\nSaved: {train_path}")
            print(f"Saved: {test_path}")


def main():
  
    import sys
    sys.path.append('src')
    from load_data import DataLoader
    
    print("Loading data...")
    loader = DataLoader()
    train_df, test_df = loader.load_parquet_files()
    
    if train_df is not None:
        print("\n\n" + "="*80)
        print("PROCESSING FULL PIPELINE (ALL TASKS COMBINED)")
        print("="*80 + "\n")
        

        preprocessor = SupplyChainPreprocessor(train_df, test_df, task="all")
        train_proc, test_proc = preprocessor.run_all_above()
        preprocessor.save_processed_data()
        
        return True
    else:
        print("Failed to load data!")
        return False


if __name__ == "__main__":
    main()
