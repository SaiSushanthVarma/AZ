
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class CostPredictionModel:   
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
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            return gpu_available
        except:
            return False

    def prepare_data(self):
 
        # Load preprocessed data which is saved in csv from preprocessing.py
        train_df = pd.read_csv('data/train_processed_cost.csv')
        self.test_df = pd.read_csv('data/test_processed_cost.csv')

        datetime_cols = train_df.select_dtypes(include=['object', 'datetime64']).columns
        if len(datetime_cols) > 0:
            train_df = train_df.drop(columns=datetime_cols)
            self.test_df = self.test_df.drop(columns=datetime_cols)
        
        # Load original data to get target
        train_original = pd.read_parquet('data/pharma_supply_chain_train.parquet')
        
        # Get target variable
        self.y_full = train_original['realistic_total_cost'].values
        self.X_full = train_df.values
        self.X_test = self.test_df.values
        
        print(f"Full training: {self.X_full.shape[0]:,}")
        print(f"Features: {self.X_full.shape[1]}")
        print(f"Test samples: {self.X_test.shape[0]:,}")
        
        print(f"\nTarget statistics:")
        print(f"  Mean: ${np.mean(self.y_full):,.2f}")
        print(f"  Median: ${np.median(self.y_full):,.2f}")
        print(f"  Std: ${np.std(self.y_full):,.2f}")
        print(f"  Min: ${np.min(self.y_full):,.2f}")
        print(f"  Max: ${np.max(self.y_full):,.2f}")
        
        # Split for validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"\nTrain split (80%): {self.X_train.shape[0]:,}")
        print(f"Validation (20%): {self.X_val.shape[0]:,}")
        
        return self
    
    def calculate_metrics(self, y_true, y_pred, name=""):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE 
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        r2 = r2_score(y_true, y_pred)
        
        if name:
            print(f"\n{name} Metrics:")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE:  ${mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}
    
    ###################### Tuning models with RandomizedSearchCV ######################
    def tune_random_forest(self):        
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
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
        
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best CV RMSE: ${np.sqrt(-random_search.best_score_):,.2f}")
        
        # Validate on holdout
        best_model = random_search.best_estimator_
        y_val_pred = best_model.predict(self.X_val)
        metrics = self.calculate_metrics(self.y_val, y_val_pred, "Validation")
        
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
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        
        if self.use_gpu:
            xgb_model = xgb.XGBRegressor(random_state=42, tree_method='gpu_hist', device='cuda:0')
        else:
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best CV RMSE: ${np.sqrt(-random_search.best_score_):,.2f}")
        
        best_model = random_search.best_estimator_
        y_val_pred = best_model.predict(self.X_val)
        metrics = self.calculate_metrics(self.y_val, y_val_pred, "Validation")
        
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
            'num_leaves': [20, 31, 40],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_samples': [10, 20, 30]
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
        
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best CV RMSE: ${np.sqrt(-random_search.best_score_):,.2f}")
        
        best_model = random_search.best_estimator_
        y_val_pred = best_model.predict(self.X_val)
        metrics = self.calculate_metrics(self.y_val, y_val_pred, "Validation")
        
        self.models['LightGBM'] = best_model
        self.best_params['LightGBM'] = random_search.best_params_
        self.results['LightGBM'] = metrics
        self.results['LightGBM']['cv_rmse'] = np.sqrt(-random_search.best_score_)
        
        return self
    ##############################################################################
    def select_best_model(self):

        for name, res in self.results.items():
            print(f"{name:<15} ${res['cv_rmse']:<14,.2f} ${res['rmse']:<14,.2f} ${res['mae']:<14,.2f}")
        
        # Select best by validation RMSE
        best_name = min(self.results.items(), key=lambda x: x[1]['rmse'])[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest Model: {best_name}")
        print(f"Best hyperparameters: {self.best_params[best_name]}")
        print(f"Validation RMSE: ${self.results[best_name]['rmse']:,.2f}")
        
        return self
    
    def retrain_on_full_data(self):

        print(f"Retraining {self.best_model_name} with best parameters on full dataset...")
        
        # Get best model type and retrain with best params
        if self.best_model_name == 'RandomForest':
            final_model = RandomForestRegressor(**self.best_params[self.best_model_name],
                                               random_state=42, n_jobs=-1)
        elif self.best_model_name == 'XGBoost':
            #final_model = xgb.XGBRegressor(**self.best_params[self.best_model_name],
            #                              random_state=42, n_jobs=-1)
            if self.use_gpu:
                final_model = xgb.XGBRegressor(
                    **self.best_params[self.best_model_name],
                    random_state=42, tree_method='gpu_hist', device='cuda:0'
                )
            else:
                final_model = xgb.XGBRegressor(
                    **self.best_params[self.best_model_name],
                    random_state=42, n_jobs=-1
                )
        else:
            #final_model = lgb.LGBMRegressor(**self.best_params[self.best_model_name],
            #                               random_state=42, n_jobs=-1, verbose=-1)
            if self.use_gpu:
                final_model = lgb.LGBMRegressor(
                    **self.best_params[self.best_model_name],
                    random_state=42, device='gpu', verbose=-1
                )
            else:
                final_model = lgb.LGBMRegressor(
                    **self.best_params[self.best_model_name],
                    random_state=42, n_jobs=-1, verbose=-1
                )
        # Train on full dataset
        final_model.fit(self.X_full, self.y_full)
        
        self.final_model = final_model
        print(f"Final model trained on {self.X_full.shape[0]:,} samples")
        
        return self
    
    def generate_predictions(self):
        # Predict on test set
        test_pred = self.final_model.predict(self.X_test)
        
        # Get order_ids from original test data
        test_original = pd.read_parquet('data/pharma_supply_chain_test.parquet')
        
        # Create submission
        submission = pd.DataFrame({
            'order_id': test_original['order_id'],
            'prediction': test_pred
        })
        
        # Save
        output_file = 'outputs/predictions/prediction_task_b_cost.csv'
        submission.to_csv(output_file, index=False)
        
        print(f"Predictions saved: {output_file}")
        print(f"\nPrediction statistics:")
        print(f"  Mean: ${test_pred.mean():,.2f}")
        print(f"  Median: ${np.median(test_pred):,.2f}")
        print(f"  Min: ${test_pred.min():,.2f}")
        print(f"  Max: ${test_pred.max():,.2f}")
        
        return submission
    
    def save_model(self):
  
        model_file = f'outputs/models/task_b_cost_{self.best_model_name.lower()}_final.pkl'
        joblib.dump(self.final_model, model_file)
        print(f"\nFinal model saved: {model_file}")
        
        return self

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
                'validation_mape': results['mape'],
                'validation_r2': results['r2'],
                'cv_rmse': results['cv_rmse'],
                'best_params': self.best_params[model_name]
            }

        import json
        with open('outputs/metrics/task_b_cost_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)

        print("Metrics saved: outputs/metrics/task_b_cost_metrics.json")
        return self

def main():

    import os
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    # Initialize
    regressor = CostPredictionModel()
    
    # Run pipeline
    regressor.prepare_data()
    regressor.tune_random_forest()
    regressor.tune_xgboost()
    regressor.tune_lightgbm()
    regressor.select_best_model()
    regressor.save_metrics() 
    regressor.retrain_on_full_data()
    regressor.generate_predictions()
    regressor.save_model()
    

    return regressor


if __name__ == "__main__":
    model = main()