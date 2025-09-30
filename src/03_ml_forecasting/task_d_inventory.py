
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class InventoryOptimizationModel:
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.use_gpu = self._check_gpu()
        print(f"GPU Available: {self.use_gpu}")

    
    def _check_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def prepare_data(self):
        
        train_df = pd.read_csv('data/train_processed_inventory.csv')
        self.test_df = pd.read_csv('data/test_processed_inventory.csv')
        
        # Drop datetime columns
        datetime_cols = train_df.select_dtypes(include=['object', 'datetime64']).columns
        if len(datetime_cols) > 0:
            train_df = train_df.drop(columns=datetime_cols)
            self.test_df = self.test_df.drop(columns=datetime_cols)
        
        train_orig = pd.read_parquet('data/pharma_supply_chain_train.parquet')
        
        self.X_full = train_df.values
        self.y_full = train_orig['optimal_stock_days'].values
        self.X_test = self.test_df.values
        
        print(f"Full training: {self.X_full.shape[0]:,}")
        print(f"Features: {self.X_full.shape[1]}")
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        return self
    
    def business_cost_function(self, y_true, y_pred):
        overstock = np.maximum(0, y_pred - y_true)
        understock = np.maximum(0, y_true - y_pred)
        holding_cost = overstock * 1.0
        stockout_cost = understock * 3.0
        return (holding_cost + stockout_cost).mean()
    
    def calculate_metrics(self, y_true, y_pred, name=""):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        business_cost = self.business_cost_function(y_true, y_pred)
        
        if name:
            print(f"\n{name} Metrics:")
        print(f"  RMSE: {rmse:.2f} days")
        print(f"  MAE:  {mae:.2f} days")
        print(f"  Business Cost: {business_cost:.2f}")
        
        return {'rmse': rmse, 'mae': mae, 'business_cost': business_cost}
    
    def tune_random_forest(self):

        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(self.X_val)
        metrics = self.calculate_metrics(self.y_val, y_pred, "Validation")
        
        self.models['RandomForest'] = best_model
        self.best_params['RandomForest'] = random_search.best_params_
        self.results['RandomForest'] = metrics
        self.results['RandomForest']['cv_rmse'] = np.sqrt(-random_search.best_score_)
        
        return self
    
    def tune_xgboost(self):
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        if self.use_gpu:
            xgb_model = xgb.XGBRegressor(random_state=42, tree_method='gpu_hist', gpu_id=0)
        else:
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(self.X_val)
        metrics = self.calculate_metrics(self.y_val, y_pred, "Validation")
        
        self.models['XGBoost'] = best_model
        self.best_params['XGBoost'] = random_search.best_params_
        self.results['XGBoost'] = metrics
        self.results['XGBoost']['cv_rmse'] = np.sqrt(-random_search.best_score_)
        
        return self
    
    def tune_lightgbm(self):

        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40]
        }
        
        if self.use_gpu:
            lgb_model = lgb.LGBMRegressor(random_state=42, device='gpu', verbose=-1)
        else:
            lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        random_search = RandomizedSearchCV(
            lgb_model, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(self.X_val)
        metrics = self.calculate_metrics(self.y_val, y_pred, "Validation")
        
        self.models['LightGBM'] = best_model
        self.best_params['LightGBM'] = random_search.best_params_
        self.results['LightGBM'] = metrics
        self.results['LightGBM']['cv_rmse'] = np.sqrt(-random_search.best_score_)
        
        return self
    
    def select_best_model(self):

        for name, res in self.results.items():
            print(f"{name:<15} Business Cost: {res['business_cost']:.2f}")
        
        best_name = min(self.results.items(), key=lambda x: x[1]['business_cost'])[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest Model: {best_name}")
        return self
    
    def retrain_on_full_data(self):

        if self.best_model_name == 'RandomForest':
            final_model = RandomForestRegressor(**self.best_params[self.best_model_name],
                                               random_state=42, n_jobs=-1)
        elif self.best_model_name == 'XGBoost':
            if self.use_gpu:
                final_model = xgb.XGBRegressor(**self.best_params[self.best_model_name],
                                              random_state=42, tree_method='gpu_hist', gpu_id=0)
            else:
                final_model = xgb.XGBRegressor(**self.best_params[self.best_model_name],
                                              random_state=42, n_jobs=-1)
        else:
            if self.use_gpu:
                final_model = lgb.LGBMRegressor(**self.best_params[self.best_model_name],
                                               random_state=42, device='gpu', verbose=-1)
            else:
                final_model = lgb.LGBMRegressor(**self.best_params[self.best_model_name],
                                               random_state=42, n_jobs=-1, verbose=-1)
        
        final_model.fit(self.X_full, self.y_full)
        self.final_model = final_model
        print("Final model trained")
        return self
    
    def generate_predictions(self):

        test_pred = self.final_model.predict(self.X_test)
        test_orig = pd.read_parquet('data/pharma_supply_chain_test.parquet')
        
        submission = pd.DataFrame({
            'order_id': test_orig['order_id'],
            'prediction': test_pred
        })
        
        submission.to_csv('outputs/predictions/prediction_task_d_inventory.csv', index=False)
        print("Predictions saved")
        
        joblib.dump(self.final_model, f'outputs/models/task_d_inventory_{self.best_model_name.lower()}_final.pkl')
        return submission

    def save_metrics(self):

        metrics_data = {
            'best_model': self.best_model_name,
            'best_params': self.best_params[self.best_model_name],
            'model_results': {}
        }

        for model_name, results in self.results.items():
            metrics_data['model_results'][model_name] = {
                'validation_rmse': results['rmse'],
                'validation_mae': results['mae'],
                'validation_business_cost': results['business_cost'],
                'cv_rmse': results['cv_rmse'],
                'best_params': self.best_params[model_name]
            }

        import json
        with open('outputs/metrics/task_d_inventory_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)

        print("Metrics saved: outputs/metrics/task_d_inventory_metrics.json")
        return self

def main():
    import os
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True) 
    
    model = InventoryOptimizationModel()
    model.prepare_data()
    model.tune_random_forest()
    model.tune_xgboost()
    model.tune_lightgbm()
    model.select_best_model()
    model.save_metrics()
    model.retrain_on_full_data()
    model.generate_predictions()

    return model


if __name__ == "__main__":
    main()