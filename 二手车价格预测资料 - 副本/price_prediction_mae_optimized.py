import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass  # 如果没有中文字体，忽略错误

# 读取数据
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print("数据加载完成，开始数据预处理...")

# 数据预处理函数
def preprocess_data(data, is_train=True):
    # 复制数据以避免修改原始数据
    df = data.copy()
    
    # 处理notRepairedDamage字段中的'-'值
    if 'notRepairedDamage' in df.columns:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
        df['notRepairedDamage'] = df['notRepairedDamage'].astype(float)
    
    # 填充缺失值
    # 对于数值型特征，使用中位数填充（对MAE更鲁棒）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # 对于类别型特征，使用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 处理regDate（注册日期）特征
    if 'regDate' in df.columns:
        # 将regDate转换为年份
        df['regYear'] = df['regDate'].astype(str).str[:4].astype(float)
        df['regMonth'] = df['regDate'].astype(str).str[4:6].astype(float)
        df.drop('regDate', axis=1, inplace=True)
    
    # 处理creatDate（创建日期）特征
    if 'creatDate' in df.columns:
        # 将creatDate转换为年份
        df['creatYear'] = df['creatDate'].astype(str).str[:4].astype(float)
        df['creatMonth'] = df['creatDate'].astype(str).str[4:6].astype(float)
        df.drop('creatDate', axis=1, inplace=True)
    
    # 计算车辆使用年限
    if 'regYear' in df.columns and 'creatYear' in df.columns:
        df['age'] = df['creatYear'] - df['regYear']
    
    # 如果是训练数据，移除目标变量
    if is_train and 'price' in df.columns:
        y = df['price']
        X = df.drop(['price', 'SaleID', 'name'], axis=1)  # 移除ID和name列
        return X, y
    else:
        # 测试数据，保存SaleID用于提交
        sale_ids = df['SaleID']
        X = df.drop(['SaleID', 'name'], axis=1)  # 移除ID和name列
        return X, sale_ids

# 预处理训练数据和测试数据
X_train, y_train = preprocess_data(train_data, is_train=True)
X_test, sale_ids = preprocess_data(test_data, is_train=False)

print(f"训练数据形状: {X_train.shape}")
print(f"测试数据形状: {X_test.shape}")

# 划分训练集和验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("开始训练以MAE为损失函数的XGBoost模型...")

# 训练以MAE为损失函数的XGBoost模型
# 注意：XGBoost中的reg:absoluteerror目标函数直接优化MAE
# 调整参数以加快训练速度
model = xgb.XGBRegressor(
    objective='reg:absoluteerror',  # 使用MAE作为损失函数进行优化
    eval_metric='mae',              # 使用MAE作为评估指标
    n_estimators=200,               # 减少树的数量以加快训练
    learning_rate=0.1,              # 增加学习率
    max_depth=6,                    # 适当减少树的最大深度
    subsample=0.8,                  # 每棵树使用的样本比例
    colsample_bytree=0.8,           # 每棵树使用的特征比例
    reg_alpha=0.1,                  # L1正则化，对MAE优化有帮助
    reg_lambda=1,                   # L2正则化
    random_state=42                 # 随机种子，保证结果可复现
)

# 训练模型
# 注意：根据XGBoost版本调整参数
model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    verbose=100                    # 每100轮打印一次日志
    # 有些XGBoost版本中early_stopping_rounds可能需要在其他位置设置
)

print("模型训练完成，开始评估模型性能...")

# 在验证集上进行预测
y_val_pred = model.predict(X_val)

# 计算MAE
mae = mean_absolute_error(y_val, y_val_pred)
print(f"验证集MAE: {mae:.2f}")

# 计算MSE和RMSE用于参考
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
print(f"验证集MSE: {mse:.2f}")
print(f"验证集RMSE: {rmse:.2f}")

# 绘制预测值与实际值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title(f'XGBoost模型预测效果 (MAE={mae:.2f})')
plt.tight_layout()
plt.savefig('prediction_scatter_mae.png')
print("预测效果散点图已保存为prediction_scatter_mae.png")

# 特征重要性分析
feature_importance = model.feature_importances_
feature_names = X_train.columns

# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# 按重要性排序
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# 绘制前20个重要特征
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('以MAE为损失函数的XGBoost模型特征重要性')
plt.tight_layout()
plt.savefig('feature_importance_mae.png')
print("特征重要性图已保存为feature_importance_mae.png")

print("\n特征重要性前10名:")
print(feature_importance_df.head(10))

# 对测试数据进行预测
print("\n开始预测测试集价格...")
test_predictions = model.predict(X_test)

# 创建提交文件
# 使用指定格式保存提交文件，使用tab分隔，保持原始数值精度
with open('xgb_price_predictions_mae.csv', 'w', encoding='utf-8') as f:
    # 写入标题行，使用tab分隔
    f.write('SaleID\tprice\n')
    # 写入数据行
    for i in range(len(test_predictions)):
        # 测试集的SaleID从200000开始
        f.write(f"{200000 + i}\t{test_predictions[i]}\n")

print("测试集预测结果已保存为xgb_price_predictions_mae.csv")

print("\n二手车价格预测完成！")
print(f"预测价格的统计信息:")
print(f"- 平均值: {test_predictions.mean():.2f}")
print(f"- 中位数: {np.median(test_predictions):.2f}")
print(f"- 最小值: {test_predictions.min():.2f}")
print(f"- 最大值: {test_predictions.max():.2f}")