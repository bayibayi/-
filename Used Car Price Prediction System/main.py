import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model import ModelBuilder


def main():
    # =================================================
    # 1. 初始化处理器和构建器
    # =================================================
    processor = DataProcessor()
    model_builder = ModelBuilder(random_state=42)

    # =================================================
    # 2. 数据加载和处理
    # =================================================
    print("数据加载和处理...")

    # 加载数据
    train_df, test_df = processor.load_data(
        'used_car_train_20200313.csv',
        'used_car_testA_20200313.csv'
    )

    # 处理测试集
    test_df['notRepairedDamage'] = test_df['notRepairedDamage'].replace('-', 'missing')
    test_df['power'] = test_df['power'].clip(10, 400)

    # 移除训练集异常值
    train_df_clean = processor.remove_outliers(train_df)

    # =================================================
    # 3. 特征工程
    # =================================================
    print("特征工程...")

    train_df_features = processor.create_features(train_df_clean)
    test_df_features = processor.create_features(test_df)

    # =================================================
    # 4. 数据分割
    # =================================================
    print("数据分割...")

    X_full = train_df_features.drop('price', axis=1)
    y_full = train_df_features['price']

    X_train, X_val, y_train, y_val = model_builder.prepare_split(X_full, y_full)

    # =================================================
    # 5. 特征编码
    # =================================================
    print("特征编码...")

    X_train_encoded, X_val_encoded, X_test_encoded = processor.encode_features(
        X_train, X_val, test_df_features
    )

    # =================================================
    # 6. 模型训练
    # =================================================
    print("模型训练...")

    # 合并训练集和验证集
    X_cv = pd.concat([X_train_encoded, X_val_encoded], ignore_index=True)
    y_cv = pd.concat([y_train, y_val], ignore_index=True)

    # 使用交叉验证确定最佳迭代次数
    best_iterations = model_builder.train_with_cv(X_cv, y_cv, n_splits=3)

    # 训练最终模型
    model_builder.train_final_model(X_cv, y_cv, n_estimators=best_iterations)

    # =================================================
    # 7. 生成预测
    # =================================================
    print("生成预测...")

    # 生成原始预测
    test_pred = model_builder.predict(X_test_encoded)

    # 简单校准
    train_mean = y_cv.mean()
    train_std = y_cv.std()
    train_min = y_cv.min()
    train_max = y_cv.max()

    test_pred_normalized = (test_pred - np.mean(test_pred)) / np.std(test_pred)
    test_pred_calibrated = test_pred_normalized * train_std + train_mean

    # 截断到合理范围
    test_pred_final = np.clip(test_pred_calibrated, train_min, train_max)
    test_pred_final = test_pred_final.round().astype(int)

    # =================================================
    # 8. 保存结果
    # =================================================
    print("保存结果...")

    test_sale_ids = processor.get_test_sale_ids()
    submit = pd.DataFrame({
        'SaleID': test_sale_ids.values,
        'price': test_pred_final
    })

    submit.to_csv('prediction_result.csv', index=False)
    print(f"✅ 预测结果已保存到: prediction_result.csv")
    print(f"提交文件大小: {len(submit)} 行")
    print(f"SaleID范围: {submit['SaleID'].min()} - {submit['SaleID'].max()}")


if __name__ == "__main__":
    main()