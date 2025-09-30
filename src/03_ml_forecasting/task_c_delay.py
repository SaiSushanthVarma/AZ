
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class DelayPredictionModel:
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
        
        train_df = pd.read_csv('data/train_processed_delay.csv')
        self.test_df = pd.read_csv('data/test_processed_delay.csv')
        
        # Drop datetime/object columns
        datetime_cols = train_df.select_dtypes(include=['object', 'datetime64']).columns
        if len(datetime_cols) > 0:
            print(f"Dropping {len(datetime_cols)} non-numeric columns")
            train_df = train_df.drop(columns=datetime_cols)
            self.test_df = self.test_df.drop(columns=datetime_cols)
        
        train_orig = pd.read_parquet('data/pharma_supply_chain_train.parquet')
        
        self.X_full = train_df.values
        self.y_full = train_orig['will_be_delayed'].values
        self.X_test = self.test_df.values
        
        print(f"Full training: {self.X_full.shape[0]:,}")
        print(f"Features: {self.X_full.shape[1]}")
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42, stratify=self.y_full
        )
        
        return self
    ############################## tune models ##############################
    def tune_logistic_regression(self):

        param_dist = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],     
            'solver': ['lbfgs']
        }
        
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(roc_auc_score)
        
        random_search = RandomizedSearchCV(
            lr, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_proba = best_model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_proba)
        
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        
        self.models['LogisticRegression'] = best_model
        self.best_params['LogisticRegression'] = random_search.best_params_
        self.results['LogisticRegression'] = {'auc': auc, 'cv_auc': random_search.best_score_}
        
        return self
    
    def tune_random_forest(self):

        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(roc_auc_score)
        
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_proba = best_model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_proba)
        
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        
        self.models['RandomForest'] = best_model
        self.best_params['RandomForest'] = random_search.best_params_
        self.results['RandomForest'] = {'auc': auc, 'cv_auc': random_search.best_score_}
        
        return self
    
    def tune_xgboost(self):

        scale_pos_weight = (self.y_train==0).sum() / (self.y_train==1).sum()
        
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'scale_pos_weight': [scale_pos_weight]
        }
        
        if self.use_gpu:
            xgb_model = xgb.XGBClassifier(random_state=42, tree_method='gpu_hist', device='cuda:0')
        else:
            xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(roc_auc_score)

        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_proba = best_model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_proba)
        
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        
        self.models['XGBoost'] = best_model
        self.best_params['XGBoost'] = random_search.best_params_
        self.results['XGBoost'] = {'auc': auc, 'cv_auc': random_search.best_score_}
        
        return self
    
    def tune_lightgbm(self):

        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40]
        }
        
        if self.use_gpu:
            lgb_model = lgb.LGBMClassifier(random_state=42, device='gpu', verbose=-1)
        else:
            lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(roc_auc_score)
        random_search = RandomizedSearchCV(
            lgb_model, param_dist, n_iter=5, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_proba = best_model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_proba)
        
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        
        self.models['LightGBM'] = best_model
        self.best_params['LightGBM'] = random_search.best_params_
        self.results['LightGBM'] = {'auc': auc, 'cv_auc': random_search.best_score_}
        
        return self
    ##########################################################################################
    def select_best_model(self):

        for name, res in self.results.items():
            print(f"{name:<20} Val AUC: {res['auc']:.4f}")
        
        best_name = max(self.results.items(), key=lambda x: x[1]['auc'])[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest Model: {best_name}")
        return self
    
    def retrain_on_full_data(self):

        if self.best_model_name == 'LogisticRegression':
            final_model = LogisticRegression(
                **self.best_params[self.best_model_name],
                max_iter=1000, random_state=42, n_jobs=-1
            )
        elif self.best_model_name == 'RandomForest':
            final_model = RandomForestClassifier(
                **self.best_params[self.best_model_name],
                random_state=42, n_jobs=-1
            )
        elif self.best_model_name == 'XGBoost':
            if self.use_gpu:
                final_model = xgb.XGBClassifier(
                    **self.best_params[self.best_model_name],
                    random_state=42, tree_method='gpu_hist', device='cuda:0'
                )
            else:
                final_model = xgb.XGBClassifier(
                    **self.best_params[self.best_model_name],
                    random_state=42, n_jobs=-1
                )
        else:
            if self.use_gpu:
                final_model = lgb.LGBMClassifier(
                    **self.best_params[self.best_model_name],
                    random_state=42, device='gpu', verbose=-1
                )
            else:
                final_model = lgb.LGBMClassifier(
                    **self.best_params[self.best_model_name],
                    random_state=42, n_jobs=-1, verbose=-1
                )
        
        print(f"Training {self.best_model_name} on {self.X_full.shape[0]:,} samples...")
        final_model.fit(self.X_full, self.y_full)
        
        self.final_model = final_model
        return self
    
    def generate_predictions(self):

        test_pred = self.final_model.predict(self.X_test)
        test_orig = pd.read_parquet('data/pharma_supply_chain_test.parquet')
        
        submission = pd.DataFrame({
            'order_id': test_orig['order_id'],
            'prediction': test_pred
        })
        
        submission.to_csv('outputs/predictions/prediction_task_c_delay.csv', index=False)
        print("Saved: outputs/predictions/prediction_task_c_delay.csv")
        
        joblib.dump(self.final_model, f'outputs/models/task_c_delay_{self.best_model_name.lower()}_final.pkl')
        print(f"Saved: outputs/models/task_c_delay_{self.best_model_name.lower()}_final.pkl")
        
        return submission
    
    def save_metrics(self):

        metrics_data = {
            'best_model': self.best_model_name,
            'best_params': self.best_params[self.best_model_name],
            'model_results': {}
        }

        for model_name, results in self.results.items():
            metrics_data['model_results'][model_name] = {
                'validation_auc': results['auc'],
                'cv_auc': results['cv_auc'],
                'best_params': self.best_params[model_name]
            }

        import json
        with open('outputs/metrics/task_c_delay_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)

        print("Metrics saved: outputs/metrics/task_c_delay_metrics.json")
        return self

def main():
    import os
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    model = DelayPredictionModel()
    model.prepare_data()
    model.tune_logistic_regression()
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