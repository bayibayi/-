import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    def __init__(self):
        self.test_sale_ids = None

    def load_data(self, train_path, test_path):
        """加载数据"""
        print("读取数据...")
        train_df = pd.read_csv(train_path, sep=' ')
        test_df = pd.read_csv(test_path, sep=' ')

        # 保存测试集的SaleID用于最终提交
        self.test_sale_ids = test_df['SaleID'].copy()

        # 删除无关列
        drop_cols = ['SaleID', 'name', 'seller', 'offerType']
        train_df.drop(columns=drop_cols, inplace=True)
        test_df.drop(columns=drop_cols, inplace=True)

        print(f"训练集大小: {train_df.shape}")
        print(f"测试集大小: {test_df.shape}")

        return train_df, test_df

    def remove_outliers(self, train_df):
        """移除异常值"""
        print("处理异常值...")

        # 处理价格异常值 - 使用1%-99%分位数
        price_q01 = train_df['price'].quantile(0.01)
        price_q99 = train_df['price'].quantile(0.99)
        print(f"原始价格范围: {train_df['price'].min()} - {train_df['price'].max()}")
        print(f"1%分位数: {price_q01}, 99%分位数: {price_q99}")

        # 筛选价格在合理范围内的样本
        train_df = train_df[(train_df['price'] >= price_q01) & (train_df['price'] <= price_q99)].copy()
        print(f"价格筛选后训练集大小: {train_df.shape}")

        # 处理功率异常值
        train_df['power'] = train_df['power'].clip(10, 400)

        # 处理notRepairedDamage异常值
        train_df['notRepairedDamage'] = train_df['notRepairedDamage'].replace('-', 'missing')

        print(f"异常值处理后训练集大小: {train_df.shape}")

        return train_df

    def create_features(self, df):
        """创建特征"""
        df = df.copy()

        # 日期特征
        df['regDate'] = pd.to_datetime(df['regDate'], format='%Y%m%d', errors='coerce')
        df['creatDate'] = pd.to_datetime(df['creatDate'], format='%Y%m%d', errors='coerce')

        # 车龄特征
        df['car_age_days'] = (df['creatDate'] - df['regDate']).dt.days
        df['car_age_days'] = df['car_age_days'].fillna(0).clip(0, 365 * 20)
        df['car_age_years'] = df['car_age_days'] / 365

        # 删除原始日期列
        df.drop(columns=['regDate', 'creatDate'], inplace=True)

        # 变换特征
        df['power_log'] = np.log1p(df['power'])
        df['kilometer_log'] = np.log1p(df['kilometer'])

        # 交互特征
        df['power_per_km'] = df['power'] / (df['kilometer'] + 1)

        # 处理分类特征
        cat_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('missing').astype(str)

        return df

    def encode_features(self, X_train, X_val, X_test):
        """编码分类特征"""
        cat_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']

        for col in cat_cols:
            if col in X_train.columns:
                # 处理缺失值
                X_train[col] = X_train[col].fillna('missing')
                X_val[col] = X_val[col].fillna('missing')
                X_test[col] = X_test[col].fillna('missing')

                # LabelEncoder
                le = LabelEncoder()
                all_categories = pd.concat([X_train[col], X_val[col]], ignore_index=True).unique()
                le.fit(list(all_categories))

                X_train[col] = le.transform(X_train[col])
                X_val[col] = X_val[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        return X_train, X_val, X_test

    def get_test_sale_ids(self):
        """获取测试集SaleID"""
        return self.test_sale_ids