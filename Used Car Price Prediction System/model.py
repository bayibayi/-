import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class ModelBuilder:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.final_model = None

    def prepare_split(self, X, y, test_size=0.2):
        """准备数据分割"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=True
        )

        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
        return X_train, X_val, y_train, y_val

    def train_with_cv(self, X, y, n_splits=3):
        """使用交叉验证训练"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
            'n_estimators': 2000
        }

        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{n_splits}")

            X_tr = X.iloc[train_idx]
            X_vl = X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_vl = y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            best_iterations.append(model.best_iteration_)

        avg_iterations = int(np.mean(best_iterations))
        print(f"平均最佳迭代次数: {avg_iterations}")

        return avg_iterations

    def train_final_model(self, X, y, n_estimators=1000):
        """训练最终模型"""
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
            'n_estimators': n_estimators
        }

        print("训练最终模型...")
        self.final_model = lgb.LGBMRegressor(**params)
        self.final_model.fit(X, y)

        return self.final_model

    def predict(self, X):
        """预测"""
        if self.final_model is None:
            raise ValueError("请先训练模型")

        return self.final_model.predict(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        print(f"MAE: {mae:.2f}")
        return mae