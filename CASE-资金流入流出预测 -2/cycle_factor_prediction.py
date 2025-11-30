import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置中文字体支持（如果需要可视化）
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 读取数据
def load_data():
    print("开始读取用户余额表数据...")
    # 读取用户余额表
    df_balance = pd.read_csv('user_balance_table.csv')
    print(f"数据读取完成，总行数：{len(df_balance)}")
    print(f"数据时间范围：{df_balance['report_date'].min()} 到 {df_balance['report_date'].max()}")
    return df_balance

# 数据预处理
def preprocess_data(df_balance):
    print("开始数据预处理...")
    # 按日期聚合申购和赎回金额
    df_daily = df_balance.groupby('report_date').agg({
        'total_purchase_amt': 'sum',
        'total_redeem_amt': 'sum'
    }).reset_index()
    
    # 将report_date转换为日期类型
    df_daily['report_date'] = pd.to_datetime(df_daily['report_date'], format='%Y%m%d')
    
    # 添加时间范围过滤：只使用2014-03-01到2014-08-31的数据
    start_date = '2014-03-01'
    end_date = '2014-08-31'
    df_daily = df_daily[(df_daily['report_date'] >= start_date) & 
                       (df_daily['report_date'] <= end_date)]
    print(f"过滤后时间范围：{start_date} 至 {end_date}，共{len(df_daily)}天数据")
    
    # 提取星期几（0=周一，6=周日）
    df_daily['weekday'] = df_daily['report_date'].dt.weekday
    
    # 提取月份中的日期
    df_daily['day_of_month'] = df_daily['report_date'].dt.day
    
    # 添加周末标识
    df_daily['is_weekend'] = (df_daily['weekday'] >= 5).astype(int)
    
    # 添加月初月末标识
    df_daily['is_month_start'] = (df_daily['day_of_month'] <= 3).astype(int)
    df_daily['is_month_end'] = (df_daily['day_of_month'] >= 28).astype(int)
    
    # 添加节假日标识
    holidays = [
        '2014-01-01', '2014-01-31', '2014-02-01', '2014-02-02', '2014-02-03', '2014-02-04', '2014-02-05',
        '2014-04-05', '2014-04-06', '2014-04-07', '2014-05-01', '2014-05-02', '2014-05-03',
        '2014-06-02', '2014-09-08', '2014-09-09', '2014-09-10', '2014-10-01', '2014-10-02',
        '2014-10-03', '2014-10-04', '2014-10-05', '2014-10-06', '2014-10-07'
    ]
    holidays_dt = pd.to_datetime(holidays)
    df_daily['is_holiday'] = df_daily['report_date'].isin(holidays_dt).astype(int)
    
    # 计算移动平均特征（趋势特征）
    df_daily['purchase_7d_ma'] = df_daily['total_purchase_amt'].rolling(window=7).mean()
    df_daily['redeem_7d_ma'] = df_daily['total_redeem_amt'].rolling(window=7).mean()
    df_daily['purchase_30d_ma'] = df_daily['total_purchase_amt'].rolling(window=30).mean()
    df_daily['redeem_30d_ma'] = df_daily['total_redeem_amt'].rolling(window=30).mean()
    
    # 填充缺失值
    df_daily = df_daily.fillna(method='bfill').fillna(method='ffill')
    
    # 添加时间权重，给最近的数据更高权重
    total_days = len(df_daily)
    df_daily['weight'] = np.linspace(0.5, 1.5, total_days)
    
    print(f"数据聚合完成，日度数据行数：{len(df_daily)}")
    print(f"新增特征：周末标识、月初月末标识、节假日标识、移动平均特征")
    return df_daily

# 计算周期因子
def calculate_cycle_factors(df_daily):
    print("开始计算周期因子...")
    
    # 使用加权平均计算基准值，给最近数据更高权重
    base_purchase = np.average(df_daily['total_purchase_amt'], weights=df_daily['weight'])
    base_redeem = np.average(df_daily['total_redeem_amt'], weights=df_daily['weight'])
    
    print(f"加权基准值 - 申购：{base_purchase:.2f}，赎回：{base_redeem:.2f}")
    
    # 计算星期因子（加权）
    weekday_factors_purchase = {}
    weekday_factors_redeem = {}
    
    for weekday in range(7):
        weekday_data = df_daily[df_daily['weekday'] == weekday]
        if len(weekday_data) > 0:
            # 使用加权平均计算星期因子
            weighted_avg_purchase = np.average(weekday_data['total_purchase_amt'], weights=weekday_data['weight'])
            weighted_avg_redeem = np.average(weekday_data['total_redeem_amt'], weights=weekday_data['weight'])
            weekday_factors_purchase[weekday] = weighted_avg_purchase / base_purchase
            weekday_factors_redeem[weekday] = weighted_avg_redeem / base_redeem
        else:
            weekday_factors_purchase[weekday] = 1.0
            weekday_factors_redeem[weekday] = 1.0
    
    print("星期因子计算完成：")
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for i, wd in enumerate(weekdays):
        print(f"{wd}: 申购因子={weekday_factors_purchase[i]:.4f}, 赎回因子={weekday_factors_redeem[i]:.4f}")
    
    # 计算月度日期因子
    day_factors_purchase = {}
    day_factors_redeem = {}
    
    for day in range(1, 32):
        day_data = df_daily[df_daily['day_of_month'] == day]
        if len(day_data) > 0:
            # 使用加权平均计算日期因子
            weighted_avg_purchase = np.average(day_data['total_purchase_amt'], weights=day_data['weight'])
            weighted_avg_redeem = np.average(day_data['total_redeem_amt'], weights=day_data['weight'])
            day_factors_purchase[day] = weighted_avg_purchase / base_purchase
            day_factors_redeem[day] = weighted_avg_redeem / base_redeem
        else:
            # 如果某一天没有数据，使用1.0作为默认因子
            day_factors_purchase[day] = 1.0
            day_factors_redeem[day] = 1.0
    
    # 计算特殊时期因子
    print("计算特殊时期因子...")
    
    # 周末因子
    weekend_data = df_daily[df_daily['is_weekend'] == 1]
    weekday_data = df_daily[df_daily['is_weekend'] == 0]
    weekend_factor_purchase = np.average(weekend_data['total_purchase_amt'], weights=weekend_data['weight']) / \
                            np.average(weekday_data['total_purchase_amt'], weights=weekday_data['weight'])
    weekend_factor_redeem = np.average(weekend_data['total_redeem_amt'], weights=weekend_data['weight']) / \
                           np.average(weekday_data['total_redeem_amt'], weights=weekday_data['weight'])
    
    # 月初因子
    month_start_data = df_daily[df_daily['is_month_start'] == 1]
    month_start_factor_purchase = np.average(month_start_data['total_purchase_amt'], weights=month_start_data['weight']) / base_purchase
    month_start_factor_redeem = np.average(month_start_data['total_redeem_amt'], weights=month_start_data['weight']) / base_redeem
    
    # 月末因子
    month_end_data = df_daily[df_daily['is_month_end'] == 1]
    month_end_factor_purchase = np.average(month_end_data['total_purchase_amt'], weights=month_end_data['weight']) / base_purchase
    month_end_factor_redeem = np.average(month_end_data['total_redeem_amt'], weights=month_end_data['weight']) / base_redeem
    
    # 节假日因子
    holiday_data = df_daily[df_daily['is_holiday'] == 1]
    non_holiday_data = df_daily[df_daily['is_holiday'] == 0]
    holiday_factor_purchase = 1.0
    holiday_factor_redeem = 1.0
    if len(holiday_data) > 0 and len(non_holiday_data) > 0:
        holiday_factor_purchase = np.average(holiday_data['total_purchase_amt'], weights=holiday_data['weight']) / \
                                np.average(non_holiday_data['total_purchase_amt'], weights=non_holiday_data['weight'])
        holiday_factor_redeem = np.average(holiday_data['total_redeem_amt'], weights=holiday_data['weight']) / \
                               np.average(non_holiday_data['total_redeem_amt'], weights=non_holiday_data['weight'])
    
    print(f"特殊时期因子：")
    print(f"周末因子 - 申购: {weekend_factor_purchase:.4f}, 赎回: {weekend_factor_redeem:.4f}")
    print(f"月初因子 - 申购: {month_start_factor_purchase:.4f}, 赎回: {month_start_factor_redeem:.4f}")
    print(f"月末因子 - 申购: {month_end_factor_purchase:.4f}, 赎回: {month_end_factor_redeem:.4f}")
    print(f"节假日因子 - 申购: {holiday_factor_purchase:.4f}, 赎回: {holiday_factor_redeem:.4f}")
    
    # 获取最新的移动平均值作为趋势参考
    latest_purchase_7d_ma = df_daily['purchase_7d_ma'].iloc[-1]
    latest_redeem_7d_ma = df_daily['redeem_7d_ma'].iloc[-1]
    latest_purchase_30d_ma = df_daily['purchase_30d_ma'].iloc[-1]
    latest_redeem_30d_ma = df_daily['redeem_30d_ma'].iloc[-1]
    
    # 计算趋势因子（使用最近7天和30天移动平均的比值）
    trend_factor_purchase = latest_purchase_7d_ma / latest_purchase_30d_ma
    trend_factor_redeem = latest_redeem_7d_ma / latest_redeem_30d_ma
    
    print(f"趋势因子 - 申购: {trend_factor_purchase:.4f}, 赎回: {trend_factor_redeem:.4f}")
    
    return {
        'base_purchase': base_purchase,
        'base_redeem': base_redeem,
        'weekday_factors_purchase': weekday_factors_purchase,
        'weekday_factors_redeem': weekday_factors_redeem,
        'day_factors_purchase': day_factors_purchase,
        'day_factors_redeem': day_factors_redeem,
        'weekend_factor_purchase': weekend_factor_purchase,
        'weekend_factor_redeem': weekend_factor_redeem,
        'month_start_factor_purchase': month_start_factor_purchase,
        'month_start_factor_redeem': month_start_factor_redeem,
        'month_end_factor_purchase': month_end_factor_purchase,
        'month_end_factor_redeem': month_end_factor_redeem,
        'holiday_factor_purchase': holiday_factor_purchase,
        'holiday_factor_redeem': holiday_factor_redeem,
        'trend_factor_purchase': trend_factor_purchase,
        'trend_factor_redeem': trend_factor_redeem,
        'latest_purchase_7d_ma': latest_purchase_7d_ma,
        'latest_redeem_7d_ma': latest_redeem_7d_ma
    }

# 预测2014-09-01到2014-09-30的数据
def predict_september(factors):
    print("开始预测2014年9月份数据...")
    
    # 创建9月份的日期序列
    september_dates = pd.date_range(start='2014-09-01', end='2014-09-30')
    
    # 2014年节假日列表
    holidays = [
        '2014-09-08', '2014-09-09', '2014-09-10'  # 中秋节相关假期
    ]
    holidays_dt = pd.to_datetime(holidays)
    
    predictions = []
    
    # 使用最新的移动平均值作为基准，而不是简单的历史平均值
    base_purchase = factors['latest_purchase_7d_ma']
    base_redeem = factors['latest_redeem_7d_ma']
    
    print(f"使用最新移动平均值作为基准 - 申购：{base_purchase:.2f}，赎回：{base_redeem:.2f}")
    
    for date in september_dates:
        # 提取基本特征
        weekday = date.weekday()
        day = date.day
        is_weekend = 1 if weekday >= 5 else 0
        is_month_start = 1 if day <= 3 else 0
        is_month_end = 1 if day >= 28 else 0
        is_holiday = 1 if date in holidays_dt else 0
        
        # 获取基本因子
        wd_factor_purchase = factors['weekday_factors_purchase'][weekday]
        wd_factor_redeem = factors['weekday_factors_redeem'][weekday]
        day_factor_purchase = factors['day_factors_purchase'].get(day, 1.0)
        day_factor_redeem = factors['day_factors_redeem'].get(day, 1.0)
        
        # 特殊时期调整因子
        special_adjustment_purchase = 1.0
        special_adjustment_redeem = 1.0
        
        # 应用节假日因子
        if is_holiday:
            special_adjustment_purchase *= factors['holiday_factor_purchase']
            special_adjustment_redeem *= factors['holiday_factor_redeem']
        # 应用月初因子
        elif is_month_start:
            special_adjustment_purchase *= factors['month_start_factor_purchase']
            special_adjustment_redeem *= factors['month_start_factor_redeem']
        # 应用月末因子
        elif is_month_end:
            special_adjustment_purchase *= factors['month_end_factor_purchase']
            special_adjustment_redeem *= factors['month_end_factor_redeem']
        # 应用周末因子（如果不是节假日的周末）
        elif is_weekend:
            special_adjustment_purchase *= factors['weekend_factor_purchase']
            special_adjustment_redeem *= factors['weekend_factor_redeem']
        
        # 使用改进的乘法模型进行预测，包含趋势因子
        predicted_purchase = base_purchase * wd_factor_purchase * day_factor_purchase * special_adjustment_purchase * factors['trend_factor_purchase']
        predicted_redeem = base_redeem * wd_factor_redeem * day_factor_redeem * special_adjustment_redeem * factors['trend_factor_redeem']
        
        # 格式化为YYYYMMDD格式的日期字符串
        date_str = date.strftime('%Y%m%d')
        
        predictions.append({
            'report_date': date_str,
            'purchase': predicted_purchase,
            'redeem': predicted_redeem
        })
    
    # 转换为DataFrame
    df_predictions = pd.DataFrame(predictions)
    print(f"预测完成，共{len(df_predictions)}天数据")
    return df_predictions

# 保存结果
def save_results(df_predictions, output_file='factor_result2.csv'):
    print(f"保存结果到{output_file}...")
    # 保存结果，不保留索引
    df_predictions.to_csv(output_file, index=False, float_format='%.2f')
    print("结果保存完成！")
    # 显示前几行结果
    print("\n预测结果前5行：")
    print(df_predictions.head())

# 主函数
def main():
    try:
        # 加载数据
        df_balance = load_data()
        
        # 数据预处理
        df_daily = preprocess_data(df_balance)
        
        # 计算周期因子
        factors = calculate_cycle_factors(df_daily)
        
        # 预测9月份数据
        df_predictions = predict_september(factors)
        
        # 保存结果
        save_results(df_predictions)
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"程序执行出错：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()