
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')


class RiskClassificationModel:
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
            
        train_df = pd.read_csv('data/train_processed_risk.csv')
        self.test_df = pd.read_csv('data/test_processed_risk.csv')
        
        # Drop datetime columns
        datetime_cols = train_df.select_dtypes(include=['object', 'datetime64']).columns
        if len(datetime_cols) > 0:
            print(f"Dropping {len(datetime_cols)} non-numeric columns")
            train_df = train_df.drop(columns=datetime_cols)
            self.test_df = self.test_df.drop(columns=datetime_cols)
        
        train_orig = pd.read_parquet('data/pharma_supply_chain_train.parquet')
        
        X_full = train_df.values
        y_full = train_orig['supply_chain_disruption_risk'].values
        
        # Encode target
        target_map = {'Low_Risk': 0, 'Medium_Risk': 1, 'High_Risk': 2}
        y_full = np.array([target_map[y] for y in y_full])
        
        # Split training data for train, val 
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )
        
        self.X_full = X_full
        self.y_full = y_full
        
        print(f"Full training: {X_full.shape[0]:,}")
        print(f"Train split (80%): {self.X_train.shape[0]:,}")
        print(f"Validation (20%): {self.X_val.shape[0]:,}")
        print(f"Features: {X_full.shape[1]}")
        
        return self
    ## required because some classes are very low on data so using SMOTE to oversample
    def handle_imbalance(self):
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_bal, self.y_train_bal = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"After SMOTE: {self.X_train_bal.shape[0]:,} samples")
        return self
###################### Tuning models with RandomizedSearchCV ######################
    def tune_random_forest(self):        
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='weighted')
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=2, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train_bal, self.y_train_bal)
        
        best_model = random_search.best_estimator_
        y_val_pred = best_model.predict(self.X_val)
        f1_val = f1_score(self.y_val, y_val_pred, average='weighted')
        
        print(f"\nBest CV F1: {random_search.best_score_:.4f}")
        print(f"Validation F1: {f1_val:.4f}")
        
        self.models['RandomForest'] = best_model
        self.best_params['RandomForest'] = random_search.best_params_
        self.results['RandomForest'] = {'f1_val': f1_val, 'f1_cv': random_search.best_score_}
        
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
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                tree_method='gpu_hist',
                device='cuda:0',
                eval_metric='mlogloss'
            )
        else:
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                n_jobs=-1, 
                eval_metric='mlogloss'
            )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='weighted')
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=2, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train_bal, self.y_train_bal)
        
        best_model = random_search.best_estimator_
        y_val_pred = best_model.predict(self.X_val)
        f1_val = f1_score(self.y_val, y_val_pred, average='weighted')
        
        print(f"\nBest CV F1: {random_search.best_score_:.4f}")
        print(f"Validation F1: {f1_val:.4f}")
        
        self.models['XGBoost'] = best_model
        self.best_params['XGBoost'] = random_search.best_params_
        self.results['XGBoost'] = {'f1_val': f1_val, 'f1_cv': random_search.best_score_}
        
        return self
    
    def tune_lightgbm(self):
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40]
        }
        
        if self.use_gpu:
            lgb_model = lgb.LGBMClassifier(
                random_state=42,
                device='gpu',
                verbose=-1
            )
        else:
            lgb_model = lgb.LGBMClassifier(
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
            )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='weighted')
        random_search = RandomizedSearchCV(
            lgb_model, param_dist, n_iter=2, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(self.X_train_bal, self.y_train_bal)
        
        best_model = random_search.best_estimator_
        y_val_pred = best_model.predict(self.X_val)
        f1_val = f1_score(self.y_val, y_val_pred, average='weighted')
        
        print(f"\nBest CV F1: {random_search.best_score_:.4f}")
        print(f"Validation F1: {f1_val:.4f}")
        
        self.models['LightGBM'] = best_model
        self.best_params['LightGBM'] = random_search.best_params_
        self.results['LightGBM'] = {'f1_val': f1_val, 'f1_cv': random_search.best_score_}
        
        return self
#################################################################

    def select_best_model(self):
        
        for name, res in self.results.items():
            print(f"{name:<15} Val F1: {res['f1_val']:.4f}")
        
        best_name = max(self.results.items(), key=lambda x: x[1]['f1_val'])[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest Model: {best_name}")
        return self
    
    def retrain_on_full_data(self):
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_full_bal, y_full_bal = smote.fit_resample(self.X_full, self.y_full)
        
        if self.best_model_name == 'RandomForest':
            final_model = RandomForestClassifier(
                **self.best_params[self.best_model_name],
                random_state=42, n_jobs=-1, class_weight='balanced'
            )
        elif self.best_model_name == 'XGBoost':
            if self.use_gpu:
                final_model = xgb.XGBClassifier(
                    **self.best_params[self.best_model_name],
                    random_state=42, tree_method='gpu_hist',
                    device='cuda:0', eval_metric='mlogloss'
                )
            else:
                final_model = xgb.XGBClassifier(
                    **self.best_params[self.best_model_name],
                    random_state=42, n_jobs=-1, eval_metric='mlogloss'
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
        
        print(f"Training {self.best_model_name} on {X_full_bal.shape[0]:,} samples...")
        final_model.fit(X_full_bal, y_full_bal)
        
        self.final_model = final_model
        print("Complete")
        return self
    
    def generate_predictions(self):
        X_test = self.test_df.values
        test_pred = self.final_model.predict(X_test)
        
        label_map = {0: 'Low_Risk', 1: 'Medium_Risk', 2: 'High_Risk'}
        test_pred_labels = [label_map[p] for p in test_pred]
        
        test_orig = pd.read_parquet('data/pharma_supply_chain_test.parquet')
        
        submission = pd.DataFrame({
            'order_id': test_orig['order_id'],
            'prediction': test_pred_labels
        })
        
        submission.to_csv('outputs/predictions/prediction_task_a_risk.csv', index=False)
        print("Saved: outputs/predictions/prediction_task_a_risk.csv")
        
        joblib.dump(self.final_model, f'outputs/models/task_a_risk_{self.best_model_name.lower()}_final.pkl')
        print(f"Saved: outputs/models/task_a_risk_{self.best_model_name.lower()}_final.pkl")
        
        return submission
    def save_metrics(self):
        metrics_data = {
            'best_model': self.best_model_name,
            'best_params': self.best_params[self.best_model_name],
            'model_results': {}
        }

        for model_name, results in self.results.items():
            metrics_data['model_results'][model_name] = {
                'validation_f1': results['f1_val'],
                'cv_f1': results['f1_cv'],
                'best_params': self.best_params[model_name]
            }

        import json
        with open('outputs/metrics/task_a_risk_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)

        print("Metrics saved: outputs/metrics/task_a_risk_metrics.json")
        return self

def main():
    import os
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    classifier = RiskClassificationModel()
    classifier.prepare_data()
    classifier.handle_imbalance()
    classifier.tune_random_forest()
    classifier.tune_xgboost()
    classifier.tune_lightgbm()
    classifier.select_best_model()
    classifier.save_metrics()
    classifier.retrain_on_full_data()
    classifier.generate_predictions()

    return classifier

if __name__ == "__main__":
    model = main()
