# 二手车价格预测系统

## 项目简介

基于机器学习的二手车价格预测系统，使用LightGBM模型对二手车价格进行预测。系统通过数据清洗、特征工程、模型训练和结果校准等步骤，生成二手车价格预测结果。
数据集来自阿里天池二手车价格预测学习赛，比赛网址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281915.0.0.5cad1262J3Wujq

## 项目结构

```
car_price_prediction/
├── main.py              # 主程序入口
├── data_processor.py    # 数据处理模块
├── model.py             # 模型构建模块
├── used_car_train_20200313.csv    # 训练数据
├── used_car_testA_20200313.csv    # 测试数据
├── prediction_result.csv          # 预测结果文件
└── README.md           # 项目说明文档
```

## 功能模块

### 1. DataProcessor（数据处理模块）
- **数据加载**：读取训练集和测试集CSV文件
- **异常值处理**：移除价格异常值和功率异常值
- **特征工程**：创建日期特征、车龄特征、对数变换特征、交互特征等
- **特征编码**：对分类特征进行Label Encoding编码

### 2. ModelBuilder（模型构建模块）
- **数据分割**：将数据集分割为训练集和验证集
- **交叉验证**：使用K折交叉验证确定最佳迭代次数
- **模型训练**：训练LightGBM回归模型
- **预测评估**：生成预测结果并计算MAE指标

### 3. Main（主程序）
- **流程控制**：协调数据处理和模型训练的完整流程
- **预测校准**：对预测结果进行统计校准
- **结果保存**：生成最终的预测结果文件

## 使用方法

### 快速开始

1. 确保数据文件在项目目录中：
   - `used_car_train_20200313.csv`
   - `used_car_testA_20200313.csv`

2. 运行主程序：
```bash
python main.py
```

3. 查看预测结果：
   - 预测结果将保存到 `prediction_result.csv` 文件中


## 输出文件

### prediction_result.csv
包含两列：
- `SaleID`：车辆ID（150000-199999）
- `price`：预测的二手车价格（整数）

示例：
```
SaleID,price
150000,12500
150001,8900
150002,15600
...
199999,7200
```
