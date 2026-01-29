"""
从给定板块中选择预测股票涨幅最好的前三个股票
同时考虑参数准确性进行优化
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import holidays
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import platform
import time

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

# API请求时间间隔控制
API_DELAY = 0.5  # API请求间隔，单位秒
last_request_time = 0  # 记录上次请求时间


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
    global last_request_time
    # 控制API请求频率
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < API_DELAY:
        time.sleep(API_DELAY - time_since_last_request)
    last_request_time = time.time()

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


def find_best_parameters_for_single_stock(df, prediction_days=10):
    """
    为单个股票找到最佳参数组合
    """
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

    # 分割数据为训练和验证部分
    split_point = len(df) - prediction_days
    train_df = df[:split_point]
    actual_future = df[split_point:]

    best_rmse = float('inf')
    best_params = None
    best_pred_df = None

    for params in param_combinations:
        try:
            # 使用训练数据进行预测
            pred_df = predict_with_params(
                train_df,
                prediction_days,
                model=None,  # 模型稍后传入
                tokenizer=None,  # 分词器稍后传入
                T=params["T"],
                top_p=params["top_p"],
                sample_count=params["sample_count"]
            )

            # 评估预测准确性
            mae, mse, rmse = evaluate_predictions(actual_future, pred_df)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_pred_df = pred_df

        except Exception as e:
            continue

    return best_params, best_rmse, best_pred_df


def get_stocks_from_board(board_code, limit=50):
    """
    获取板块中的股票列表
    """
    global last_request_time
    # 控制API请求频率
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < API_DELAY:
        time.sleep(API_DELAY - time_since_last_request)
    last_request_time = time.time()

    try:
        # 获取板块成分股
        stock_list = ak.stock_board_concept_cons_em(symbol=board_code)
        # 只取前limit个股票，避免太多影响性能
        return stock_list['代码'].tolist()[:limit]
    except Exception as e:
        print(f"获取板块 {board_code} 成分股失败: {e}")
        return []


def predict_top_stocks_in_board(board_code, top_n=3, prediction_days=10):
    """
    在指定板块中预测并选出涨幅最好的前N个股票
    同时考虑参数准确性
    """
    print(f"开始分析板块 {board_code} 中股票未来{prediction_days}天的预测涨幅...")

    # 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 获取板块成分股
    print(f"正在获取板块 {board_code} 的成分股...")
    stock_codes = get_stocks_from_board(board_code)

    if not stock_codes:
        print(f"未能获取板块 {board_code} 的成分股")
        return

    print(f"获取到 {len(stock_codes)} 只股票")

    # 存储预测结果
    predictions = []

    for i, code in enumerate(stock_codes):
        print(f"正在处理第 {i + 1}/{len(stock_codes)} 只股票: {code}")

        # 添加小延迟以避免过于频繁的请求
        time.sleep(0.1)

        try:
            # 获取股票数据
            df = get_stock_data(code, days=300)
            if df.empty:
                print(f"  - 获取股票 {code} 数据为空，跳过")
                continue

            df = df.set_index('timestamps')

            # 为这只股票找到最佳参数
            best_params, best_rmse, _ = find_best_parameters_for_single_stock(df, prediction_days)

            if best_params is None:
                print(f"  - 股票 {code} 参数优化失败，跳过")
                continue

            # 使用最佳参数进行实际预测
            pred_df = predict_with_params(
                df,  # 使用完整数据进行最终预测
                prediction_days,
                model,
                tokenizer,
                T=best_params["T"],
                top_p=best_params["top_p"],
                sample_count=best_params["sample_count"]
            )

            # 计算涨幅
            current_price = df['close'].iloc[-1]
            future_price = pred_df['close'].iloc[-1]
            growth_rate = (future_price - current_price) / current_price

            # 计算置信度分数（基于参数准确性和涨幅）
            confidence_score = calculate_confidence_score(growth_rate, best_rmse, best_params)

            predictions.append({
                'code': code,
                'name': get_stock_name(code),  # 获取股票名称
                'current_price': current_price,
                'predicted_price': future_price,
                'growth_rate': growth_rate,
                'best_params': best_params,
                'rmse': best_rmse,
                'confidence_score': confidence_score
            })

            print(f"  - 股票 {code}: 当前价格 {current_price:.2f}, 预测价格 {future_price:.2f}, "
                  f"涨幅 {growth_rate * 100:.2f}%, RMSE {best_rmse:.4f}, 置信度 {confidence_score:.4f}")

        except Exception as e:
            print(f"  - 处理股票 {code} 时出错: {e}")
            continue

    if not predictions:
        print("没有成功预测任何股票")
        return

    # 按置信度分数排序，选择前top_n个
    sorted_predictions = sorted(predictions, key=lambda x: x['confidence_score'], reverse=True)
    top_predictions = sorted_predictions[:top_n]

    print(f"\n=== 板块 {board_code} 中预测涨幅前{top_n}的股票 ===")
    for i, item in enumerate(top_predictions, 1):
        print(f"{i}. {item['name']} ({item['code']})")
        print(f"   当前价格: {item['current_price']:.2f}")
        print(f"   预测价格: {item['predicted_price']:.2f}")
        print(f"   预计涨幅: {item['growth_rate'] * 100:.2f}%")
        print(f"   参数: T={item['best_params']['T']}, top_p={item['best_params']['top_p']}, "
              f"sample_count={item['best_params']['sample_count']}")
        print(f"   RMSE: {item['rmse']:.4f}")
        print(f"   置信度: {item['confidence_score']:.4f}")
        print()

    return top_predictions


def calculate_confidence_score(growth_rate, rmse, params):
    """
    计算综合置信度分数，考虑涨幅和预测准确性
    """
    # 基础置信度：涨幅越高，基础分数越高
    base_score = abs(growth_rate)

    # 准确性惩罚：RMSE越大，惩罚越大
    accuracy_penalty = rmse / 10  # 归一化处理

    # 参数稳定性奖励：较保守的参数（T较小，top_p适中）给予一定偏好
    param_stability_bonus = 0
    if params['T'] <= 1.0 and 0.8 <= params['top_p'] <= 0.95:
        param_stability_bonus = 0.05

    # 综合分数
    confidence_score = base_score - accuracy_penalty + param_stability_bonus

    return confidence_score


def get_stock_name(code):
    """
    获取股票名称
    """
    global last_request_time
    # 控制API请求频率
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < API_DELAY:
        time.sleep(API_DELAY - time_since_last_request)
    last_request_time = time.time()

    try:
        # 获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=code)
        if not stock_info.empty:
            return stock_info['股票简称'].iloc[0]
        else:
            return f"股票{code}"
    except:
        return f"股票{code}"


def predict_top_stocks_by_board(board_code, top_n=3):
    """
    对外接口：预测指定板块中涨幅前N的股票
    """
    return predict_top_stocks_in_board(board_code, top_n=top_n)


if __name__ == "__main__":
    # 示例：选择一个板块代码进行测试
    board_code = 'BK0427'  # 示例板块代码，你可以替换成任何有效的板块代码

    # 预测板块中涨幅前3的股票
    top_stocks = predict_top_stocks_by_board(board_code, top_n=3)

    if top_stocks:
        print("预测完成！以下是涨幅前3的股票：")
        for i, stock in enumerate(top_stocks, 1):
            print(f"{i}. {stock['name']} ({stock['code']}) - 预计涨幅: {stock['growth_rate'] * 100:.2f}%")
    else:
        print("预测失败，没有获得任何结果")
