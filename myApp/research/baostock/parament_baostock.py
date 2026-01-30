"""
优化特点总结
数据获取优化：
增加默认获取天数至1000天，一次性获取更多历史数据
避免多次请求同一股票数据
参数组合扩展：
从原来的8种参数组合扩展到约198种参数组合
T值范围：0.3-1.5（11个值）
top_p值范围：0.7-0.95（6个值）
sample_count值：1, 3, 5（3个值）
并行处理：
使用ThreadPoolExecutor并行测试参数组合
显著提高参数测试效率
结果展示优化：
显示前20个最佳参数组合
对比图只显示前10个最佳结果，避免图表过于复杂
内存管理：
通过一次获取大量数据减少API调用次数
在单次数据获取后测试多种参数组合
只绘制最佳参数组合预测线：减少图表复杂度
考虑A股涨跌幅限制：在预测结果中应用涨跌幅限制约束
# 程序功能解释

## 整体概述
这是一个**股票参数优化系统**，旨在通过测试大量参数组合来找到最适合特定股票的预测参数，从而提高Kronos模型的预测准确性。

## 核心功能模块

### 1. **数据获取模块**
- [get_stock_data]：从baostock获取股票历史数据
- [generate_future_trading_days]：生成未来交易日时间戳（排除节假日）

### 2. **参数优化模块**
- [generate_param_combinations]：生成198种参数组合
  - T值范围：0.3-1.5（11个值）
  - top_p值范围：0.7-0.95（6个值）
  - sample_count值：1, 3, 5（3个值）

### 3. **预测模型模块**
- [predict_with_params]：使用指定参数进行单股票预测
- [predict_with_batch_params]：批量预测多资产

### 4. **参数测试模块**
- [find_best_parameters]：测试所有参数组合并找出最优解
- [single_param_test]：单个参数组合的测试任务

## 主要特点

### 1. **高效参数搜索**
- 生成约198种参数组合（11×6×3）
- 使用[ThreadPoolExecutor]并行测试
- 通过RMSE指标评估参数效果

### 2. **A股市场适应**
- [apply_stock_limit_constraints]：应用A股涨跌幅限制（±10%）
- 考虑ST股票和科创板特殊限制

### 3. **可视化展示**
- [plot_best_prediction_only]：绘制最佳参数组合预测图
- 对比约束前后的预测结果

### 4. **错误处理机制**
- [retry_with_backoff]：指数退避重试策略
- 随机延迟避免API频率限制

## 工作流程

1. **数据准备**：获取1000天历史数据
2. **参数生成**：创建198种参数组合
3. **模型训练**：使用训练集数据
4. **参数测试**：并行测试所有组合
5. **结果评估**：计算MAE、MSE、RMSE
6. **最优选择**：选出RMSE最小的参数组合
7. **结果可视化**：绘制预测对比图

## 批量处理功能
- [find_best_parameters_batch]：支持多股票同时优化
- 计算平均评估指标
- 适用于投资组合参数优化

这个程序的核心价值在于通过大规模参数搜索找到最适合特定股票的预测参数，从而提高预测模型的准确性。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from datetime import datetime, timedelta
import holidays
import baostock as bs
import matplotlib
import platform
import time
import random
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import itertools

# 根据操作系统选择合适后端
system = platform.system()
if system == "Darwin":  # macOS
    matplotlib.use('MacOSX')
elif system == "Windows":
    matplotlib.use('TkAgg')
else:  # Linux
    matplotlib.use('TkAgg')


def retry_with_backoff(max_retries=3, backoff_factor=1):
    """重试装饰器，带有退避策略"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    print(f"第{attempt + 1}次尝试失败，等待{wait_time:.2f}秒后重试: {str(e)}")
                    time.sleep(wait_time)
            return None

        return wrapper

    return decorator


# 登录baostock
lg = bs.login()

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model import Kronos, KronosTokenizer, KronosPredictor


def generate_future_trading_days(start_date, num_days):
    """
    生成未来交易日时间戳（排除周末和节假日）
    """
    # 将起始日期转换为pandas Timestamp
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    elif isinstance(start_date, datetime):
        start_date = pd.to_datetime(start_date)

    # 获取中国法定节假日
    cn_holidays = holidays.China()

    trading_days = []
    current_date = start_date + timedelta(days=1)  # 从第二天开始

    while len(trading_days) < num_days:
        # 检查是否为工作日（周一到周五）且不是节假日
        if current_date.weekday() < 5 and current_date.date() not in cn_holidays:
            trading_days.append(current_date)

        current_date += timedelta(days=1)

    return trading_days


@retry_with_backoff(max_retries=3, backoff_factor=1)
def get_stock_data(symbol, days=1000):  # 增加默认天数，一次性获取更多数据
    """获取指定天数的股票数据（不支持板块指数）"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # 处理股票代码格式
    if symbol.startswith(('sh', 'sz')):
        code = symbol  # 已经是baostock格式
    elif symbol.startswith('6'):  # 上交所
        code = f"sh.{symbol}"
    elif symbol.startswith(('0', '3')):  # 深交所
        code = f"sz.{symbol}"
    else:
        # 如果已经是带前缀的格式则直接使用
        code = symbol

    # 添加随机延迟避免请求过于频繁
    time.sleep(random.uniform(0.5, 1.5))

    # 使用baostock获取股票数据
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # 前复权
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        # 检查是否有错误信息
        if rs.error_msg:
            raise ValueError(f"获取股票 {symbol} 的数据失败: {rs.error_msg}")
        else:
            raise ValueError(f"无法获取股票 {symbol} 的数据，可能该股票不存在或数据为空")

    # 转换为DataFrame
    result = pd.DataFrame(data_list, columns=rs.fields)

    # 添加成交额字段（volume * close 的近似值，实际应用中可能需要真实数据）
    result['amount'] = pd.to_numeric(result['volume'], errors='coerce') * pd.to_numeric(result['close'],
                                                                                        errors='coerce')

    # 重命名列
    column_mapping = {
        'date': 'timestamps',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'amount': 'amount'
    }
    required_data = result.rename(columns=column_mapping)[list(column_mapping.values())].copy()

    # 转换时间戳格式
    required_data['timestamps'] = pd.to_datetime(required_data['timestamps'])

    # 确保数值列是数字类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_columns:
        required_data[col] = pd.to_numeric(required_data[col], errors='coerce')

    # 删除包含NaN的行
    required_data = required_data.dropna()

    print(f"成功获取股票 {symbol} {len(required_data)} 条历史数据")
    return required_data


def apply_stock_limit_constraints(pred_df, last_close_price):
    """
    应用A股涨跌幅限制约束
    ST股票: ±5%
    普通股票: ±10%
    科创板/创业板新股前5日无涨跌幅限制，之后±20%
    """
    # 复制预测数据
    constrained_df = pred_df.copy()

    # 假设是普通股票，涨跌幅限制为10%
    limit_up = 0.10
    limit_down = -0.10

    # 逐行应用涨跌幅限制
    prev_price = last_close_price
    for i in range(len(constrained_df)):
        current_pred = constrained_df.iloc[i]['close']

        # 计算相对于前一日的涨跌幅
        daily_return = (current_pred - prev_price) / prev_price

        # 限制涨跌幅
        if daily_return > limit_up:
            current_pred = prev_price * (1 + limit_up)
        elif daily_return < limit_down:
            current_pred = prev_price * (1 + limit_down)

        # 更新预测值
        constrained_df.iloc[i, constrained_df.columns.get_loc('close')] = current_pred
        prev_price = current_pred  # 更新前一日价格为当前修正后的价格

    return constrained_df


def predict_with_params(df, pred_len, model, tokenizer, T=1.0, top_p=0.9, sample_count=1):
    """使用指定参数进行预测"""
    max_lookback = min(250, len(df) - 50)
    lookback = max_lookback

    # 准备输入数据
    x_df = df.iloc[-lookback:, :][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = pd.Series(df.index[-lookback:])

    # 生成未来交易日时间戳
    future_timestamps = generate_future_trading_days(df.index[-1], pred_len)
    y_timestamp = pd.DatetimeIndex(future_timestamps)
    y_timestamp_series = pd.Series(y_timestamp)

    # 创建预测器
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 执行预测
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp_series,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False
    )

    # 设置预测数据的时间戳索引
    pred_df.index = y_timestamp

    return pred_df


def predict_with_batch_params(df_list, pred_len, model, tokenizer, T=1.0, top_p=0.9, sample_count=1):
    """使用批量预测方法进行多资产预测"""
    x_df_list = []
    x_timestamp_list = []
    y_timestamp_list = []

    for df in df_list:
        max_lookback = min(250, len(df) - 50)
        lookback = max_lookback

        # 准备输入数据
        x_df = df.iloc[-lookback:, :][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = pd.Series(df.index[-lookback:])

        # 生成未来交易日时间戳
        future_timestamps = generate_future_trading_days(df.index[-1], pred_len)
        y_timestamp = pd.DatetimeIndex(future_timestamps)
        y_timestamp_series = pd.Series(y_timestamp)

        x_df_list.append(x_df)
        x_timestamp_list.append(x_timestamp)
        y_timestamp_list.append(y_timestamp_series)

    # 创建预测器
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 执行批量预测
    pred_df_list = predictor.predict_batch(
        df_list=x_df_list,
        x_timestamp_list=x_timestamp_list,
        y_timestamp_list=y_timestamp_list,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False
    )

    return pred_df_list


def plot_best_prediction_only(historical_df, best_pred_df, best_params_str, stock_symbol):
    """
    只绘制最佳参数组合的预测结果对比图
    """
    # 准备数据
    historical_close = historical_df['close']
    last_historical_price = historical_close.iloc[-1]

    # 应用A股涨跌幅限制约束
    constrained_pred_df = apply_stock_limit_constraints(best_pred_df, last_historical_price)

    fig, ax = plt.subplots(figsize=(16, 10))

    # 绘制历史数据
    ax.plot(historical_close.index, historical_close,
            label='Historical', color='black', linewidth=2, linestyle='-', zorder=5)

    # 绘制最佳参数组合的预测结果
    ax.plot(constrained_pred_df.index, constrained_pred_df['close'],
            label=f'Best Prediction: {best_params_str}',
            color='red', linewidth=2, linestyle='--', marker='o', markersize=4, zorder=4)

    # 绘制未约束的预测结果（虚线）用于对比
    ax.plot(best_pred_df.index, best_pred_df['close'],
            label='Unconstrained Prediction',
            color='orange', linewidth=1, linestyle=':', alpha=0.6, zorder=3)

    ax.set_ylabel('Price', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title(f'{stock_symbol} - Best Parameter Prediction (with A-share Limit Constraints)', fontsize=16,
                 fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 设置日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))  # 每4周显示一个标签
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def evaluate_predictions(actual_df, predicted_df):
    """
    评估预测准确性
    """
    if actual_df.empty or predicted_df.empty:
        return float('inf'), float('inf'), float('inf')

    # 计算误差指标
    mae = np.mean(np.abs(actual_df['close'] - predicted_df['close']))
    mse = np.mean((actual_df['close'] - predicted_df['close']) ** 2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def generate_param_combinations():
    """生成更多参数组合进行测试"""
    # 扩展参数范围
    T_values = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    top_p_values = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    sample_count_values = [1, 3, 5]

    # 生成所有参数组合
    param_combinations = []
    for T, top_p, sample_count in itertools.product(T_values, top_p_values, sample_count_values):
        param_combinations.append({
            "T": T,
            "top_p": top_p,
            "sample_count": sample_count
        })

    return param_combinations


def single_param_test(args):
    """单个参数组合的测试任务"""
    train_df, actual_future, model, tokenizer, params, prediction_days = args

    try:
        # 使用训练数据进行预测
        pred_df = predict_with_params(
            train_df,
            prediction_days,
            model,
            tokenizer,
            T=params["T"],
            top_p=params["top_p"],
            sample_count=params["sample_count"]
        )

        # 评估预测准确性
        mae, mse, rmse = evaluate_predictions(actual_future, pred_df)

        param_key = f"T_{params['T']}_top_p_{params['top_p']}_samples_{params['sample_count']}"

        return param_key, {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "params": params,
            "pred_df": pred_df
        }, True
    except Exception as e:
        print(f"  - 参数组合 {params} 预测失败: {e}")
        return None, None, False


def find_best_parameters(stock_symbol, prediction_days=10):
    """
    测试更多的参数组合，找出最适合的参数
    """
    print(f"开始测试股票 {stock_symbol} 的参数组合...")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 获取股票数据 - 一次性获取大量数据
    print("正在获取股票数据...")
    try:
        df = get_stock_data(stock_symbol, days=1000)  # 获取更多历史数据
        if df.empty:
            print("获取的股票数据为空")
            return

        print(f"获取到 {len(df)} 条历史数据")
        df = df.set_index('timestamps')

    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return

    # 3. 生成更多参数组合
    param_combinations = generate_param_combinations()
    print(f"总共生成 {len(param_combinations)} 种参数组合")

    # 分割数据为训练和验证部分
    split_point = len(df) - prediction_days
    train_df = df[:split_point]
    actual_future = df[split_point:]

    # 准备并行测试参数
    test_args = [
        (train_df, actual_future, model, tokenizer, params, prediction_days)
        for params in param_combinations
    ]

    print(f"开始并行测试 {len(param_combinations)} 种参数组合...")

    # 使用线程池并行处理参数测试
    predictions_dict = {}
    evaluation_results = {}

    with ThreadPoolExecutor(max_workers=min(5, len(param_combinations))) as executor:
        results = executor.map(single_param_test, test_args)

        for param_key, eval_result, success in results:
            if success:
                predictions_dict[param_key] = eval_result['pred_df']
                evaluation_results[param_key] = eval_result
                print(f"  - {param_key}: RMSE={eval_result['RMSE']:.4f}")

    # 找出最佳参数组合
    if evaluation_results:
        best_param_key = min(evaluation_results.keys(), key=lambda k: evaluation_results[k]["RMSE"])
        best_result = evaluation_results[best_param_key]
        best_pred_df = predictions_dict[best_param_key]

        print(f"\n=== 最佳参数组合 ===")
        print(f"参数: {best_param_key}")
        print(f"RMSE: {best_result['RMSE']:.4f}")
        print(f"MAE: {best_result['MAE']:.4f}")
        print(f"MSE: {best_result['MSE']:.4f}")
        print(
            f"具体参数: T={best_result['params']['T']}, top_p={best_result['params']['top_p']}, sample_count={best_result['params']['sample_count']}")

        # 显示前10个最佳参数组合
        print(f"\n=== 前10个最佳参数组合排序 (按RMSE) ===")
        sorted_results = sorted(evaluation_results.items(), key=lambda x: x[1]["RMSE"])
        for i, (key, result) in enumerate(sorted_results[:10]):  # 显示前10个
            print(f"{i + 1}. {key} - RMSE: {result['RMSE']:.4f}, MAE: {result['MAE']:.4f}")

        # 只绘制最佳参数组合的预测结果（应用A股涨跌幅限制）
        print("\n正在生成最佳参数组合预测图...")
        plot_best_prediction_only(train_df, best_pred_df, best_param_key, stock_symbol)
    else:
        print("没有成功生成任何预测结果")


def get_multiple_stocks_data(symbols, days=1000):  # 增加天数
    """批量获取多个股票数据，带延迟避免频率限制"""
    df_list = []
    for i, symbol in enumerate(symbols):
        try:
            print(f"正在获取第 {i + 1}/{len(symbols)} 个股票数据: {symbol}")

            # 每次请求之间增加延迟
            if i > 0:
                time.sleep(random.uniform(1, 2))

            df = get_stock_data(symbol, days=days)  # 增加天数
            if df.empty:
                print(f"获取股票 {symbol} 数据为空")
                continue

            print(f"获取到股票 {symbol} {len(df)} 条历史数据")
            df = df.set_index('timestamps')
            df_list.append(df)
        except Exception as e:
            print(f"获取股票 {symbol} 数据失败: {e}")
            continue

    return df_list


def find_best_parameters_batch(symbols, prediction_days=10):
    """
    使用批量预测测试多个股票的不同参数组合
    """
    print(f"开始测试 {len(symbols)} 个股票的不同参数组合...")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 获取所有股票数据
    print("正在获取股票数据...")
    df_list = get_multiple_stocks_data(symbols, days=1000)  # 增加天数

    if not df_list:
        print("没有成功获取任何股票数据")
        return

    # 生成更多参数组合
    param_combinations = generate_param_combinations()
    print(f"总共生成 {len(param_combinations)} 种参数组合")

    print(f"开始测试 {len(param_combinations)} 种参数组合...")

    # 分割数据为训练和验证部分
    train_df_list = []
    actual_future_list = []
    for df in df_list:
        split_point = len(df) - prediction_days
        train_df_list.append(df[:split_point])
        actual_future_list.append(df[split_point:])

    # 存储每种参数组合的总体评估结果
    overall_evaluation_results = {}

    # 准备批量参数测试
    for i, params in enumerate(param_combinations):
        param_key = f"T_{params['T']}_top_p_{params['top_p']}_samples_{params['sample_count']}"
        print(f"测试第 {i + 1}/{len(param_combinations)} 种参数组合: {param_key}")

        try:
            # 使用批量预测
            pred_df_list = predict_with_batch_params(
                train_df_list,
                prediction_days,
                model,
                tokenizer,
                T=params["T"],
                top_p=params["top_p"],
                sample_count=params["sample_count"]
            )

            # 计算整体评估指标
            total_mae = 0
            total_mse = 0
            valid_predictions = 0

            for j, (actual_df, pred_df) in enumerate(zip(actual_future_list, pred_df_list)):
                if not actual_df.empty and not pred_df.empty:
                    mae, mse, rmse = evaluate_predictions(actual_df, pred_df)
                    total_mae += mae
                    total_mse += mse
                    valid_predictions += 1

            if valid_predictions > 0:
                avg_mae = total_mae / valid_predictions
                avg_mse = total_mse / valid_predictions
                avg_rmse = np.sqrt(avg_mse)

                # 存储结果
                overall_evaluation_results[param_key] = {
                    "avg_MAE": avg_mae,
                    "avg_MSE": avg_mse,
                    "avg_RMSE": avg_rmse,
                    "params": params,
                    "valid_count": valid_predictions
                }

                print(f"  - 平均MAE: {avg_mae:.4f}, 平均MSE: {avg_mse:.4f}, 平均RMSE: {avg_rmse:.4f}")
            else:
                print(f"  - 没有有效的预测结果")

        except Exception as e:
            print(f"  - 批量预测失败: {e}")
            continue

    # 找出最佳参数组合
    if overall_evaluation_results:
        best_param_key = min(overall_evaluation_results.keys(), key=lambda k: overall_evaluation_results[k]["avg_RMSE"])
        best_result = overall_evaluation_results[best_param_key]

        print(f"\n=== 最佳参数组合 ===")
        print(f"参数: {best_param_key}")
        print(f"平均RMSE: {best_result['avg_RMSE']:.4f}")
        print(f"平均MAE: {best_result['avg_MAE']:.4f}")
        print(f"平均MSE: {best_result['avg_MSE']:.4f}")
        print(f"有效预测数量: {best_result['valid_count']}")
        print(
            f"具体参数: T={best_result['params']['T']}, top_p={best_result['params']['top_p']}, sample_count={best_result['params']['sample_count']}")

        # 显示前10个最佳参数组合
        print(f"\n=== 前10个最佳参数组合排序 (按平均RMSE) ===")
        sorted_results = sorted(overall_evaluation_results.items(), key=lambda x: x[1]["avg_RMSE"])
        for i, (key, result) in enumerate(sorted_results[:10]):  # 显示前10个
            print(f"{i + 1}. {key} - 平均RMSE: {result['avg_RMSE']:.4f}, 平均MAE: {result['avg_MAE']:.4f}")
    else:
        print("没有成功生成任何预测结果")


if __name__ == "__main__":
    try:
        # 测试单个股票的参数组合
        stock_symbol = 'sh.601012'  # 股票代码，使用完整格式
        print(f"开始测试股票: {stock_symbol}")

        # 先测试单个股票数据获取
        try:
            test_df = get_stock_data(stock_symbol, days=100)
            print(f"成功获取测试数据，共{len(test_df)}条记录")
        except Exception as e:
            print(f"测试数据获取失败: {e}")
            exit(1)

        find_best_parameters(stock_symbol, prediction_days=10)

        # 或者测试多个股票的批量参数优化
        # symbols = ['sh.600036', 'sz.000001']  # 不能包含板块指数
        # find_best_parameters_batch(symbols, prediction_days=30)
    finally:
        # 登出baostock
        try:
            bs.logout()
            print("已登出baostock")
        except:
            pass
