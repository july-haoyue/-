import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass  # 如果没有中文字体，忽略错误

# 读取数据
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print("数据加载完成，开始深入分析车型特征...")

# 基本信息分析
def analyze_model_feature():
    print("\n=== 车型(model)特征基本信息 ===")
    print(f"车型特征数据类型: {train_data['model'].dtype}")
    print(f"车型特征缺失值数量: {train_data['model'].isnull().sum()}")
    
    # 车型唯一值数量
    unique_models = train_data['model'].nunique()
    print(f"车型唯一值数量: {unique_models}")
    
    # 车型分布统计
    model_counts = train_data['model'].value_counts()
    print(f"\n车型分布前10名:")
    print(model_counts.head(10))
    
    # 车型与价格的关系分析
    model_price_analysis(model_counts)
    
    # 车型与其他关键特征的关系
    model_correlation_analysis()
    
    # 车型特征工程建议
    feature_engineering_suggestions()

# 分析车型与价格的关系
def model_price_analysis(model_counts):
    print("\n=== 车型与价格的关系分析 ===")
    
    # 计算每个车型的平均价格
    model_avg_price = train_data.groupby('model')['price'].mean().sort_values(ascending=False)
    
    print(f"平均价格最高的10个车型:")
    print(model_avg_price.head(10))
    
    print(f"\n平均价格最低的10个车型:")
    print(model_avg_price.tail(10))
    
    # 绘制车型价格分布箱线图（选择出现次数较多的车型）
    top_models = model_counts.head(20).index
    top_model_data = train_data[train_data['model'].isin(top_models)]
    
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='model', y='price', data=top_model_data)
    plt.title('主要车型价格分布箱线图')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_price_boxplot.png')
    print("\n车型价格分布箱线图已保存为model_price_boxplot.png")
    
    # 绘制车型平均价格条形图
    plt.figure(figsize=(14, 10))
    top_avg_models = model_avg_price.head(30)
    sns.barplot(x=top_avg_models.index, y=top_avg_models.values)
    plt.title('平均价格前30名车型')
    plt.xlabel('车型编号')
    plt.ylabel('平均价格')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_avg_price_top30.png')
    print("车型平均价格前30名条形图已保存为model_avg_price_top30.png")
    
    # 计算车型特征与价格的相关性
    corr_model_price = train_data['model'].corr(train_data['price'])
    print(f"\n车型特征与价格的线性相关系数: {corr_model_price:.4f}")
    
    # 统计分析
    perform_statistical_test()

# 统计显著性检验
def perform_statistical_test():
    # 选择两个代表性车型进行T检验
    model_0_data = train_data[train_data['model'] == 0]['price']
    model_100_data = train_data[train_data['model'] == 100]['price']
    
    # 确保有足够的数据进行检验
    if len(model_0_data) > 10 and len(model_100_data) > 10:
        t_stat, p_value = stats.ttest_ind(model_0_data, model_100_data)
        print(f"\n车型0和车型100价格差异的T检验结果:")
        print(f"T统计量: {t_stat:.4f}, P值: {p_value:.4f}")
        print(f"差异{'统计显著' if p_value < 0.05 else '不显著'}")

# 分析车型与其他特征的关系
def model_correlation_analysis():
    print("\n=== 车型与其他关键特征的相关性分析 ===")
    
    # 选择关键特征进行相关性分析
    key_features = ['model', 'brand', 'bodyType', 'fuelType', 'power', 'kilometer', 'price']
    correlation_matrix = train_data[key_features].corr()
    
    print("车型与其他关键特征的相关系数:")
    print(correlation_matrix['model'])
    
    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('车型与其他关键特征的相关性热力图')
    plt.tight_layout()
    plt.savefig('model_correlation_heatmap.png')
    print("\n车型相关性热力图已保存为model_correlation_heatmap.png")
    
    # 分析车型与品牌的关系
    model_brand_counts = train_data.groupby(['brand', 'model']).size().unstack(fill_value=0)
    print("\n车型与品牌的组合关系示例:")
    # 只显示前5个品牌的前5个车型
    if model_brand_counts.shape[0] >= 5 and model_brand_counts.shape[1] >= 5:
        print(model_brand_counts.iloc[:5, :5])

# 特征工程建议
def feature_engineering_suggestions():
    print("\n=== 车型特征工程建议 ===")
    print("1. 车型聚类:")
    print("   - 基于价格分布和其他属性，将相似价格水平的车型分组")
    print("   - 可以使用K-means或层次聚类算法")
    print("\n2. 车型-品牌组合特征:")
    print("   - 创建车型和品牌的组合特征，捕获品牌溢价效应")
    print("   - 例如: brand_model = brand * 1000 + model")
    print("\n3. 车型稀有度特征:")
    print("   - 计算每个车型在数据集中的出现频率")
    print("   - 稀有车型可能有不同的价格行为")
    print("\n4. 车型-年限交互特征:")
    print("   - 不同车型随使用年限的贬值率可能不同")
    print("   - 创建model_age = model * age交互特征")
    print("\n5. 车型价格等级编码:")
    print("   - 基于历史平均价格，将车型编码为低、中、高价格等级")
    print("   - 使用目标编码技术捕获车型的价格信息")
    print("\n6. 缺失值处理优化:")
    print("   - 当前使用中位数填充，可考虑根据品牌或其他相关特征的分布填充")
    
    # 创建一个示例的特征工程函数
def demo_feature_engineering():
    print("\n=== 示例：车型特征工程实现 ===")
    
    # 创建示例数据集
    demo_df = train_data[['model', 'brand', 'price']].copy()
    
    # 1. 计算车型出现频率（稀有度）
    model_counts = demo_df['model'].value_counts()
    demo_df['model_frequency'] = demo_df['model'].map(model_counts)
    demo_df['model_rarity'] = 1 / (demo_df['model_frequency'] + 1)  # 加1避免除零
    
    # 2. 创建品牌-车型组合特征
    demo_df['brand_model'] = demo_df['brand'] * 1000 + demo_df['model']
    
    # 3. 基于价格的车型编码
    model_price_mean = demo_df.groupby('model')['price'].transform('mean')
    demo_df['model_price_level'] = pd.qcut(model_price_mean, 3, labels=['low', 'medium', 'high'])
    
    print("特征工程后的数据示例:")
    print(demo_df.head())
    
    return demo_df

if __name__ == "__main__":
    # 执行完整分析
    analyze_model_feature()
    
    # 展示特征工程示例
    demo_df = demo_feature_engineering()
    
    print("\n车型特征深入分析完成！")
    print("生成的可视化文件:")
    print("1. model_price_boxplot.png - 主要车型价格分布箱线图")
    print("2. model_avg_price_top30.png - 平均价格前30名车型条形图")
    print("3. model_correlation_heatmap.png - 车型与其他特征相关性热力图")