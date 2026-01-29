import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from datetime import datetime, timedelta
import holidays
import akshare as ak
import matplotlib
import platform

# 根据操作系统选择合适后端
system = platform.system()
if system == "Darwin":  # macOS
    matplotlib.use('MacOSX')
elif system == "Windows":
    matplotlib.use('TkAgg')
else:  # Linux
    matplotlib.use('TkAgg')

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


def get_stock_data(symbol, days=500):
    """获取指定天数的股票或板块指数数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    # 判断是否为板块指数
    if symbol.startswith('BK'):
        # 使用板块指数专用接口
        stockdata = ak.stock_board_industry_hist_em(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=""  # 板块指数不需要复权
        )
    else:
        # 使用普通股票接口
        stockdata = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

    # 重命名列
    column_mapping = {
        '日期': 'timestamps',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }
    required_data = stockdata.rename(columns=column_mapping)[list(column_mapping.values())].copy()

    # 转换时间戳格式
    required_data['timestamps'] = pd.to_datetime(required_data['timestamps'])

    # 确保数值列是数字类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_columns:
        required_data[col] = pd.to_numeric(required_data[col], errors='coerce')

    # 删除包含NaN的行
    required_data = required_data.dropna()

    return required_data


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


def plot_parameter_comparison(historical_df, predictions_dict, stock_symbol):
    """
    绘制不同参数组合的预测结果对比图
    """
    # 准备数据
    historical_close = historical_df['close']

    fig, ax = plt.subplots(figsize=(16, 10))

    # 绘制历史数据
    ax.plot(historical_close.index, historical_close,
            label='Historical', color='black', linewidth=2, linestyle='-', zorder=5)

    # 绘制不同参数组合的预测结果
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (param_key, pred_df) in enumerate(predictions_dict.items()):
        if pred_df is not None and not pred_df.empty:
            color_idx = i % len(colors)
            ax.plot(pred_df.index, pred_df['close'],
                    label=f'Params: {param_key}',
                    color=colors[color_idx], linewidth=2, linestyle='--', marker='o', markersize=4, zorder=4)

    ax.set_ylabel('Price', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title(f'{stock_symbol} - Parameter Combination Comparison', fontsize=16, fontweight='bold')
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


def find_best_parameters(stock_symbol, prediction_days=10):
    """
    测试不同的参数组合，找出最适合的参数
    """
    print(f"开始测试股票 {stock_symbol} 的不同参数组合...")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 获取股票数据
    print("正在获取股票数据...")
    try:
        df = get_stock_data(stock_symbol, days=300)  # 获取较长时间的历史数据
        if df.empty:
            print("获取的股票数据为空")
            return

        print(f"获取到 {len(df)} 条历史数据")
        df = df.set_index('timestamps')

    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return

    # 定义参数组合
    param_combinations = [
        {"T": 0.5, "top_p": 0.8, "sample_count": 1},
        {"T": 0.7, "top_p": 0.85, "sample_count": 1},
        {"T": 0.8, "top_p": 0.9, "sample_count": 1},
        {"T": 1.0, "top_p": 0.9, "sample_count": 1},
        {"T": 1.2, "top_p": 0.95, "sample_count": 1},
        {"T": 0.6, "top_p": 0.8, "sample_count": 3},
        {"T": 0.8, "top_p": 0.9, "sample_count": 3},
        {"T": 1.0, "top_p": 0.9, "sample_count": 3},
    ]

    # 存储预测结果和评估指标
    predictions_dict = {}
    evaluation_results = {}

    print(f"开始测试 {len(param_combinations)} 种参数组合...")

    # 分割数据为训练和验证部分
    split_point = len(df) - prediction_days
    train_df = df[:split_point]
    actual_future = df[split_point:]

    for i, params in enumerate(param_combinations):
        param_key = f"T_{params['T']}_top_p_{params['top_p']}_samples_{params['sample_count']}"
        print(f"测试第 {i + 1}/{len(param_combinations)} 种参数组合: {param_key}")

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

            # 存储结果
            predictions_dict[param_key] = pred_df
            evaluation_results[param_key] = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "params": params
            }

            print(f"  - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        except Exception as e:
            print(f"  - 预测失败: {e}")
            continue

    # 找出最佳参数组合
    if evaluation_results:
        best_param_key = min(evaluation_results.keys(), key=lambda k: evaluation_results[k]["RMSE"])
        best_result = evaluation_results[best_param_key]

        print(f"\n=== 最佳参数组合 ===")
        print(f"参数: {best_param_key}")
        print(f"RMSE: {best_result['RMSE']:.4f}")
        print(f"MAE: {best_result['MAE']:.4f}")
        print(f"MSE: {best_result['MSE']:.4f}")
        print(
            f"具体参数: T={best_result['params']['T']}, top_p={best_result['params']['top_p']}, sample_count={best_result['params']['sample_count']}")

        # 显示所有参数组合的排序结果
        print(f"\n=== 所有参数组合排序 (按RMSE) ===")
        sorted_results = sorted(evaluation_results.items(), key=lambda x: x[1]["RMSE"])
        for i, (key, result) in enumerate(sorted_results[:10]):  # 显示前10个
            print(f"{i + 1}. {key} - RMSE: {result['RMSE']:.4f}, MAE: {result['MAE']:.4f}")

        # 绘制对比图
        print("\n正在生成参数组合对比图...")
        plot_parameter_comparison(train_df, predictions_dict, stock_symbol)
    else:
        print("没有成功生成任何预测结果")


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
    df_list = []
    for symbol in symbols:
        try:
            df = get_stock_data(symbol, days=300)  # 获取较长时间的历史数据
            if df.empty:
                print(f"获取股票 {symbol} 数据为空")
                continue

            print(f"获取到股票 {symbol} {len(df)} 条历史数据")
            df = df.set_index('timestamps')
            df_list.append(df)
        except Exception as e:
            print(f"获取股票 {symbol} 数据失败: {e}")
            continue

    if not df_list:
        print("没有成功获取任何股票数据")
        return

    # 定义参数组合
    param_combinations = [
        {"T": 0.5, "top_p": 0.8, "sample_count": 1},
        {"T": 0.7, "top_p": 0.85, "sample_count": 1},
        {"T": 0.8, "top_p": 0.9, "sample_count": 1},
        {"T": 1.0, "top_p": 0.9, "sample_count": 1},
        {"T": 1.2, "top_p": 0.95, "sample_count": 1},
        {"T": 0.6, "top_p": 0.8, "sample_count": 3},
        {"T": 0.8, "top_p": 0.9, "sample_count": 3},
        {"T": 1.0, "top_p": 0.9, "sample_count": 3},
    ]

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

        # 显示所有参数组合的排序结果
        print(f"\n=== 所有参数组合排序 (按平均RMSE) ===")
        sorted_results = sorted(overall_evaluation_results.items(), key=lambda x: x[1]["avg_RMSE"])
        for i, (key, result) in enumerate(sorted_results[:10]):  # 显示前10个
            print(f"{i + 1}. {key} - 平均RMSE: {result['avg_RMSE']:.4f}, 平均MAE: {result['avg_MAE']:.4f}")
    else:
        print("没有成功生成任何预测结果")


if __name__ == "__main__":
    # 测试单个股票的参数组合
    stock_symbol = '600682'  # 板块指数代码
    find_best_parameters(stock_symbol, prediction_days=10)

    # 或者测试多个股票的批量参数优化
    # symbols = ['BK0427', 'BK0428', '600036']  # 可以包含板块指数和股票
    # find_best_parameters_batch(symbols, prediction_days=30)
