import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class to handle loading and initial validation of pharmaceutical supply chain data"""
    
    def __init__(self, train_path='data/pharma_supply_chain_train.parquet', 
                 test_path='data/pharma_supply_chain_test.parquet'):
        
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        
    def load_parquet_files(self):
        print("Loading data from parquet files...")
        try:
            self.train_df = pd.read_parquet(self.train_path)
            self.test_df = pd.read_parquet(self.test_path)
            print(f"✓ Training data loaded: {self.train_df.shape}")
            print(f"✓ Test data loaded: {self.test_df.shape}")
            return self.train_df, self.test_df
        except Exception as e:
            print(f"Error loading parquet files: {e}")
            return None, None
    # just as a backup
    def load_csv_files(self, train_csv='data/pharma_supply_chain_train.csv',
                       test_csv='data/pharma_supply_chain_test.csv'):

        print("Loading data from CSV files...")
        try:
            self.train_df = pd.read_csv(train_csv)
            self.test_df = pd.read_csv(test_csv)
            
            # Convert datetime columns
            date_cols = ['order_date', 'order_timestamp']
            for col in date_cols:
                if col in self.train_df.columns:
                    self.train_df[col] = pd.to_datetime(self.train_df[col])
                    self.test_df[col] = pd.to_datetime(self.test_df[col])
            
            print(f"Training data loaded: {self.train_df.shape}")
            print(f"Test data loaded: {self.test_df.shape}")
            return self.train_df, self.test_df
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            return None, None
    
    def get_basic_info(self):
        if self.train_df is None:
            print("Please load data first!")
            return

        
        print("\ntraining data")
        print(f"Shape: {self.train_df.shape}")
        print(f"Memory Usage: {self.train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\ntest data")
        print(f"Shape: {self.test_df.shape}")
        print(f"Memory Usage: {self.test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\ntarget variables in training data:")
        target_vars = ['supply_chain_disruption_risk', 'realistic_total_cost', 
                      'will_be_delayed', 'optimal_stock_days', 'quality_risk_score']
        for var in target_vars:
            if var in self.train_df.columns:
                print(f"  {var} ok")
            else:
                print(f"  {var} (missing)")
        
        return {
            'train_shape': self.train_df.shape,
            'test_shape': self.test_df.shape,
            'train_cols': list(self.train_df.columns),
            'test_cols': list(self.test_df.columns)
        }
    
    def get_data_types_summary(self):
        if self.train_df is None:
            print("Please load data first!")
            return
        
        dtypes_count = self.train_df.dtypes.value_counts()
        for dtype, count in dtypes_count.items():
            print(f"  {dtype}: {count} columns")

        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  Count: {len(numeric_cols)}")
        cat_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        print(f"  Count: {len(cat_cols)}")
        date_cols = self.train_df.select_dtypes(include=['datetime64']).columns.tolist()
        print(f"  Count: {len(date_cols)}")
        print(f"  Columns: {date_cols}")
        
        return {
            'numeric_columns': numeric_cols,
            'categorical_columns': cat_cols,
            'datetime_columns': date_cols
        }
    
    def get_missing_values_summary(self):
 
        if self.train_df is None:
            print("Please load data first!")
            return
        missing = self.train_df.isnull().sum()
        missing_pct = (missing / len(self.train_df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percentage': missing_pct
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        if len(missing_df) > 0:
            print(f"\nColumns with missing values: {len(missing_df)}")
            print(missing_df.head(15))
        else:
            print("No missing values found!")
        
        return missing_df
    
    def save_to_csv(self, output_dir='data/'):

        if self.train_df is None:
            print("Please load data first!")
            return
        
        try:
            self.train_df.to_csv(f'{output_dir}pharma_supply_chain_train.csv', index=False)
            self.test_df.to_csv(f'{output_dir}pharma_supply_chain_test.csv', index=False)
            print(f"\n✓ CSV files saved to {output_dir}")
        except Exception as e:
            print(f"Error saving CSV files: {e}")


def main():
    loader = DataLoader()
    
    # Load data
    train_df, test_df = loader.load_parquet_files()
    
    if train_df is not None:
        # Get basic information
        loader.get_basic_info()
        
        # Get data types summary
        loader.get_data_types_summary()
        
        # Get missing values summary
        loader.get_missing_values_summary()
        
        print("\n" + "="*80)
        print("Data loading complete! Ready for EDA.")
        print("="*80)
        
        return train_df, test_df
    else:
        print("\nFailed to load data. Please check file paths.")
        return None, None


if __name__ == "__main__":
    train_df, test_df = main()