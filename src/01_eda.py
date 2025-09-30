
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# data loader
import sys
sys.path.append('src')
from load_data import DataLoader

class SupplyChainEDA:
    """Comprehensive EDA for pharmaceutical supply chain data"""
    
    def __init__(self, train_df, test_df):
        """
        Initialize EDA class with training and test dataframes
        Parameters: train_df, test_df
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        
        # plotting styles
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # output directory for figures
        import os
        os.makedirs('outputs/figures', exist_ok=True)
        
    def overview_statistics(self):
        print("PART 1: Dataset Overview & Target Variables identified later for ML Tasks")
                
        
        print(f"Training set: {self.train_df.shape[0]:,} rows × {self.train_df.shape[1]} columns")
        print(f"Test set: {self.test_df.shape[0]:,} rows × {self.test_df.shape[1]} columns")
        
        ## Basic info of target variables
        # ML Task A: Risk Classification
        if 'supply_chain_disruption_risk' in self.train_df.columns:
            print("\nTask A - Supply Chain Disruption Risk:")
            risk_dist = self.train_df['supply_chain_disruption_risk'].value_counts()
            risk_pct = self.train_df['supply_chain_disruption_risk'].value_counts(normalize=True) * 100
            for risk, count in risk_dist.items():
                print(f"  {risk}: {count:,} ({risk_pct[risk]:.2f}%)")
        
        # ML Task B: Total Cost
        if 'realistic_total_cost' in self.train_df.columns:
            print("\nTask B - Realistic Total Cost:")
            print(f"  Mean: ${self.train_df['realistic_total_cost'].mean():,.2f}")
            print(f"  Median: ${self.train_df['realistic_total_cost'].median():,.2f}")
            print(f"  Std Dev: ${self.train_df['realistic_total_cost'].std():,.2f}")
            print(f"  Min: ${self.train_df['realistic_total_cost'].min():,.2f}")
            print(f"  Max: ${self.train_df['realistic_total_cost'].max():,.2f}")
        
        # Task C: Delivery Delay
        if 'will_be_delayed' in self.train_df.columns:
            print("\nTask C - Delivery Delay:")
            delay_dist = self.train_df['will_be_delayed'].value_counts()
            delay_pct = self.train_df['will_be_delayed'].value_counts(normalize=True) * 100
            print(f"  On-time (0): {delay_dist.get(0, 0):,} ({delay_pct.get(0, 0):.2f}%)")
            print(f"  Delayed (1): {delay_dist.get(1, 0):,} ({delay_pct.get(1, 0):.2f}%)")
        
        # Task D: Optimal Stock Days
        if 'optimal_stock_days' in self.train_df.columns:
            print("\nTask D - Optimal Stock Days:")
            print(f"  Mean: {self.train_df['optimal_stock_days'].mean():.2f} days")
            print(f"  Median: {self.train_df['optimal_stock_days'].median():.2f} days")
            print(f"  Std Dev: {self.train_df['optimal_stock_days'].std():.2f} days")
            print(f"  Min: {self.train_df['optimal_stock_days'].min():.2f} days")
            print(f"  Max: {self.train_df['optimal_stock_days'].max():.2f} days")
    
    def data_quality_analysis(self):

        print("PART 2: DATA QUALITY ANALYSIS")
       
        print("\nMissing values")
        missing = self.train_df.isnull().sum()
        missing_pct = (missing / len(self.train_df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        print(f"\nTotal columns with missing values: {len(missing_df)}")
        print("\nTop 15 columns with missing values:")
        print(missing_df.head(15).to_string(index=False))
        
        print("\n checking for duplicates")
        train_duplicates = self.train_df.duplicated().sum()
        test_duplicates = self.test_df.duplicated().sum()
        print(f"Training set duplicates: {train_duplicates}")
        print(f"Test set duplicates: {test_duplicates}")
        
        print("\n checking for any inconsistencies")
        
        # delivery days
        if 'actual_delivery_days' in self.train_df.columns:
            neg_delivery = (self.train_df['actual_delivery_days'] < 0).sum()
            print(f"Negative actual delivery days: {neg_delivery}")
        
        # Check for impossible lead times
        if 'total_lead_time_days' in self.train_df.columns:
            neg_lead = (self.train_df['total_lead_time_days'] < 0).sum()
            print(f"Negative total lead time: {neg_lead}")
        
        # Check cost consistency
        if all(col in self.train_df.columns for col in ['total_cost_per_batch', 'realistic_total_cost']):
            cost_diff = (self.train_df['realistic_total_cost'] < self.train_df['total_cost_per_batch']).sum()
            print(f"Realistic cost < Batch cost (inconsistent): {cost_diff}")
        
        return missing_df
    
    def temporal_analysis(self):

        print("PART 3: TEMPORAL ANALYSIS")
        
        if 'order_date' not in self.train_df.columns:
            print("No temporal data available")
            return
        
        print("\n date range")
        print(f"Start date: {self.train_df['order_date'].min()}")
        print(f"End date: {self.train_df['order_date'].max()}")
        print(f"Total days: {(self.train_df['order_date'].max() - self.train_df['order_date'].min()).days}")
        
        print("\n orders by quarter")
        quarter_counts = self.train_df['quarter'].value_counts().sort_index()
        for quarter, count in quarter_counts.head(10).items():
            print(f"  {quarter}: {count:,} orders")
        
        print("\n orders by day of week")
        dow_counts = self.train_df['day_of_week'].value_counts()
        for day, count in dow_counts.items():
            print(f"  {day}: {count:,} orders")
        
        # Trend over months
        self.train_df['year_month'] = self.train_df['order_date'].dt.to_period('M')
        monthly_orders = self.train_df.groupby('year_month').size()
        
        print("\n monthly order trends")
        print(f"Average orders per month: {monthly_orders.mean():.0f}")
        print(f"Peak month: {monthly_orders.idxmax()} ({monthly_orders.max():,} orders)")
        print(f"Lowest month: {monthly_orders.idxmin()} ({monthly_orders.min():,} orders)")
    
    def product_analysis(self):
        print("PART 4: Product category & formulation analysis")       
        cat_dist = self.train_df['product_category'].value_counts()
        cat_pct = self.train_df['product_category'].value_counts(normalize=True) * 100
        for cat, count in cat_dist.items():
            print(f"  {cat}: {count:,} ({cat_pct[cat]:.2f}%)")
        
        print("\nformulation types")
        form_dist = self.train_df['formulation'].value_counts()
        form_pct = self.train_df['formulation'].value_counts(normalize=True) * 100
        for form, count in form_dist.items():
            print(f"  {form}: {count:,} ({form_pct[form]:.2f}%)")
        
        print("\n temperaure requirements")
        temp_dist = self.train_df['temperature_requirement'].value_counts()
        for temp, count in temp_dist.items():
            print(f"  {temp}: {count:,}")
        
        if 'cold_chain_required' in self.train_df.columns:
            cold_chain = self.train_df['cold_chain_required'].value_counts()
            print("\ncold chain requirements")
            print(f"  Required: {cold_chain.get(1.0, 0):,}")
            print(f"  Not Required: {cold_chain.get(0.0, 0):,}")
    
    def supply_chain_network_analysis(self):
        
        print("PART 5: Supply chain network analysis")
        print("="*80)
        
        print("\nmanufacturing sites")
        mfg_dist = self.train_df['manufacturing_site'].value_counts()
        for site, count in mfg_dist.items():
            pct = (count / len(self.train_df)) * 100
            print(f"  {site}: {count:,} ({pct:.2f}%)")
        
        print("\nDistribution centers")
        dc_dist = self.train_df['distribution_center'].value_counts()
        for dc, count in dc_dist.items():
            pct = (count / len(self.train_df)) * 100
            print(f"  {dc}: {count:,} ({pct:.2f}%)")
        
        print("\nmarket regions")
        region_dist = self.train_df['market_region'].value_counts()
        for region, count in region_dist.items():
            pct = (count / len(self.train_df)) * 100
            print(f"  {region}: {count:,} ({pct:.2f}%)")
        
        print("\nshipping modes")
        ship_dist = self.train_df['shipping_mode'].value_counts()
        for mode, count in ship_dist.items():
            pct = (count / len(self.train_df)) * 100
            print(f"  {mode}: {count:,} ({pct:.2f}%)")
    
    def regulatory_compliance_analysis(self):
        print("PART 6: Regulatory & compliance analysis")

        print("\n GMP ")
        if 'gmp_compliant' in self.train_df.columns:
            gmp_rate = (self.train_df['gmp_compliant'] == 1).sum() / len(self.train_df) * 100
            print(f"  Compliant: {gmp_rate:.2f}%")
        
        print("\n FDA ")
        if 'fda_approved' in self.train_df.columns:
            fda_rate = (self.train_df['fda_approved'] == 1).sum() / len(self.train_df) * 100
            print(f"  Approved: {fda_rate:.2f}%")
        
        print("\n EMA")
        if 'ema_approved' in self.train_df.columns:
            ema_rate = (self.train_df['ema_approved'] == 1).sum() / len(self.train_df) * 100
            print(f"  Approved: {ema_rate:.2f}%")
        
        print("\n serialization & track & trace")
        if 'serialization_required' in self.train_df.columns:
            serial_rate = (self.train_df['serialization_required'] == 1).sum() / len(self.train_df) * 100
            print(f"  Serialization required: {serial_rate:.2f}%")
        
        if 'track_trace_compliant' in self.train_df.columns:
            track_rate = (self.train_df['track_trace_compliant'] == 1).sum() / len(self.train_df) * 100
            print(f"  Track & Trace compliant: {track_rate:.2f}%")
    
    def cost_analysis(self):
        print("PART 7: Cost analysis")
                
        cost_cols = ['api_cost_per_kg', 'excipient_cost_per_kg', 'packaging_cost_per_unit',
                     'manufacturing_cost_per_batch', 'total_cost_per_batch', 'realistic_total_cost']
        
        for col in cost_cols:
            if col in self.train_df.columns:
                print(f"\n[{col}]")
                print(f"  Mean: ${self.train_df[col].mean():,.2f}")
                print(f"  Median: ${self.train_df[col].median():,.2f}")
                print(f"  Std: ${self.train_df[col].std():,.2f}")
                print(f"  Min: ${self.train_df[col].min():,.2f}")
                print(f"  Max: ${self.train_df[col].max():,.2f}")
    
    def risk_analysis(self):
        print("PART 8: Risk factor analysis")
        print("="*80)
        
        if 'supplier_risk_score' in self.train_df.columns:
            print("\n supplier risk score")
            print(f"  Mean: {self.train_df['supplier_risk_score'].mean():.2f}")
            print(f"  Median: {self.train_df['supplier_risk_score'].median():.2f}")
            print(f"  Min: {self.train_df['supplier_risk_score'].min():.2f}")
            print(f"  Max: {self.train_df['supplier_risk_score'].max():.2f}")
        
        if 'stock_out_risk' in self.train_df.columns:
            print("\n stock out risk")
            print(f"  Mean: {self.train_df['stock_out_risk'].mean():.2f}")
            print(f"  Median: {self.train_df['stock_out_risk'].median():.2f}")
        
        if 'defect_rate_ppm' in self.train_df.columns:
            print("\n defect rate (ppm)")
            print(f"  Mean: {self.train_df['defect_rate_ppm'].mean():.2f}")
            print(f"  Median: {self.train_df['defect_rate_ppm'].median():.2f}")
        
        if 'quality_risk_score' in self.train_df.columns:
            print("\n quality risk score")
            print(f"  Mean: {self.train_df['quality_risk_score'].mean():.2f}")
            print(f"  Median: {self.train_df['quality_risk_score'].median():.2f}")

    def categorical_consistency(self):
        
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_vals = self.train_df[col].nunique()
            print(f"{col}: {unique_vals} unique categories")
            if unique_vals > 20:
                print(f"  [!] High cardinality, may need encoding or grouping")

    def outlier_analysis(self):

        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}

        for col in numeric_cols:
            q1 = self.train_df[col].quantile(0.25)
            q3 = self.train_df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((self.train_df[col] < lower) | (self.train_df[col] > upper)).sum()
            if outliers > 0:
                outlier_summary[col] = outliers

        if outlier_summary:
            print("Outliers detected in:")
            for col, cnt in outlier_summary.items():
                print(f"  {col}: {cnt:,} records")
        else:
            print("No significant outliers detected.")

        return outlier_summary



    def correlation_analysis(self):
        # Analyze correlations between key numerical variables but can be done for all
       
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 1:
            corr_matrix = self.train_df[numeric_cols].corr()

            # Save full correlation matrix for later analysis 
            corr_matrix.to_csv("outputs/full_correlation_matrix.csv")
            print(f"Full correlation matrix saved to outputs/full_correlation_matrix.csv")

            # Show top correlations (absolute value)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j],
                        'abs_corr': abs(corr_matrix.iloc[i, j])
                    })

            corr_df = pd.DataFrame(corr_pairs).sort_values('abs_corr', ascending=False)

            print("\n[9.1] STRONGEST CORRELATIONS (Top 15 by |r|)")
            print(corr_df.head(15).to_string(index=False))

            # heatmap for visualization
            plt.figure(figsize=(14, 10))
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap (All Variables)")
            plt.tight_layout()
            plt.savefig("outputs/figures/correlation_heatmap.png", dpi=300)
            #print("Saved: correlation_heatmap.png")

            return corr_matrix, corr_df
        else:
            print("Not enough numeric columns for correlation analysis.")
            return None, None



    def generate_visualizations(self):
                
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Task A: Risk Classification
        if 'supply_chain_disruption_risk' in self.train_df.columns:
            risk_counts = self.train_df['supply_chain_disruption_risk'].value_counts()
            axes[0, 0].bar(risk_counts.index, risk_counts.values, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Task A: Supply Chain Disruption Risk Distribution', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Risk Level')
            axes[0, 0].set_ylabel('Count')
            for i, v in enumerate(risk_counts.values):
                axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Task B: Cost Distribution
        if 'realistic_total_cost' in self.train_df.columns:
            axes[0, 1].hist(self.train_df['realistic_total_cost'], bins=50, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Task B: Realistic Total Cost Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Cost ($)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Task C: Delivery Delay
        if 'will_be_delayed' in self.train_df.columns:
            delay_counts = self.train_df['will_be_delayed'].value_counts()
            axes[1, 0].bar(['On-time', 'Delayed'], delay_counts.values, color='coral', edgecolor='black')
            axes[1, 0].set_title('Task C: Delivery Delay Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Status')
            axes[1, 0].set_ylabel('Count')
            for i, v in enumerate(delay_counts.values):
                axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Task D: Optimal Stock Days
        if 'optimal_stock_days' in self.train_df.columns:
            axes[1, 1].hist(self.train_df['optimal_stock_days'], bins=50, color='plum', edgecolor='black')
            axes[1, 1].set_title('Task D: Optimal Stock Days Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Days')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/target_distributions.png', dpi=300, bbox_inches='tight')
        print("Saved: target_distributions.png")
        plt.close()
        
        # 2. Product Category Analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        cat_counts = self.train_df['product_category'].value_counts()
        axes[0].barh(cat_counts.index, cat_counts.values, color='steelblue')
        axes[0].set_title('Product Category Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Count')
        
        form_counts = self.train_df['formulation'].value_counts()
        axes[1].barh(form_counts.index, form_counts.values, color='darkorange')
        axes[1].set_title('Formulation Type Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Count')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/product_analysis.png', dpi=300, bbox_inches='tight')
        #print(" Saved: product_analysis.png")
        plt.close()
        
        # 3. Supply Chain Network
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        mfg_counts = self.train_df['manufacturing_site'].value_counts()
        axes[0, 0].barh(mfg_counts.index, mfg_counts.values, color='teal')
        axes[0, 0].set_title('Manufacturing Sites', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Count')
        
        dc_counts = self.train_df['distribution_center'].value_counts()
        axes[0, 1].barh(dc_counts.index, dc_counts.values, color='indianred')
        axes[0, 1].set_title('Distribution Centers', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Count')
        
        region_counts = self.train_df['market_region'].value_counts()
        axes[1, 0].bar(region_counts.index, region_counts.values, color='mediumseagreen')
        axes[1, 0].set_title('Market Regions', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Region')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        ship_counts = self.train_df['shipping_mode'].value_counts()
        axes[1, 1].bar(ship_counts.index, ship_counts.values, color='slateblue')
        axes[1, 1].set_title('Shipping Modes', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Mode')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/supply_chain_network.png', dpi=300, bbox_inches='tight')
        #print("Saved: supply_chain_network.png")
        plt.close()
        
        # 4. Temporal Analysis
        if 'order_date' in self.train_df.columns:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Monthly trend
            monthly_orders = self.train_df.groupby(self.train_df['order_date'].dt.to_period('M')).size()
            monthly_orders.index = monthly_orders.index.to_timestamp()
            axes[0].plot(monthly_orders.index, monthly_orders.values, marker='o', linewidth=2, color='navy')
            axes[0].set_title('Monthly Order Trends', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Month')
            axes[0].set_ylabel('Number of Orders')
            axes[0].grid(True, alpha=0.3)
            
            # Day of week
            dow_counts = self.train_df['day_of_week'].value_counts().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            )
            axes[1].bar(dow_counts.index, dow_counts.values, color='darkgreen')
            axes[1].set_title('Orders by Day of Week', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Day')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('outputs/figures/temporal_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved: temporal_analysis.png")
            plt.close()
        
        # 5. Cost Analysis
        cost_cols = ['api_cost_per_kg', 'excipient_cost_per_kg', 'packaging_cost_per_unit',
                     'manufacturing_cost_per_batch']
        available_cost_cols = [col for col in cost_cols if col in self.train_df.columns]
        
        if available_cost_cols:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for idx, col in enumerate(available_cost_cols[:4]):
                axes[idx].hist(self.train_df[col].dropna(), bins=50, color='gold', edgecolor='black')
                axes[idx].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Cost ($)')
                axes[idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('outputs/figures/cost_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

    
    def run_eda(self):

        self.overview_statistics()
        self.data_quality_analysis()
        self.outlier_analysis()
        self.categorical_consistency()
        self.temporal_analysis()
        self.product_analysis()
        self.supply_chain_network_analysis()
        self.regulatory_compliance_analysis()
        self.cost_analysis()
        self.risk_analysis()
        self.correlation_analysis()
        
        # Generate visualizations
        self.generate_visualizations()
        
        return True


def main():    
    # Load data using DataLoader
    loader = DataLoader()
    train_df, test_df = loader.load_parquet_files()
    
    if train_df is not None:
        # Initialize EDA
        eda = SupplyChainEDA(train_df, test_df)
        
        # Run complete EDA
        eda.run_eda()
    else:
        print("Failed to load data. Please check file paths.")


if __name__ == "__main__":
    main()