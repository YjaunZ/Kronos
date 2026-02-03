import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mplfinance.original_flavor import candlestick_ohlc
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
import gc
import uuid

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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


def get_stock_name(symbol):
    """获取股票名称"""
    # 处理股票代码格式
    if symbol.startswith(('sh', 'sz')):
        code = symbol  # 已经是baostock格式
    elif symbol.startswith('6'):  # 上交所
        code = f"sh.{symbol}"
    elif symbol.startswith(('0', '3')):  # 深交所
        code = f"sz.{symbol}"
    else:
        code = symbol

    # 查询股票基本信息
    rs = bs.query_stock_basic(code=code)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if data_list:
        stock_info = pd.DataFrame(data_list, columns=rs.fields)
        if not stock_info.empty:
            return stock_info.iloc[0]['code_name']

    # 如果无法获取股票名称，返回股票代码
    return symbol


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


def get_stock_data_weekly(symbol, weeks=200):
    """获取周线数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    # 处理股票代码格式
    if symbol.startswith(('sh', 'sz')):
        code = symbol
    elif symbol.startswith('6'):
        code = f"sh.{symbol}"
    elif symbol.startswith(('0', '3')):
        code = f"sz.{symbol}"
    else:
        code = f"sz.{symbol}"

    # 添加随机延迟避免请求过于频繁
    time.sleep(random.uniform(0.5, 1.5))

    # 使用baostock获取周线数据
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="w",
        adjustflag="3"  # 前复权
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        if rs.error_msg:
            raise ValueError(f"获取股票 {symbol} 的周线数据失败: {rs.error_msg}")
        else:
            raise ValueError(f"无法获取股票 {symbol} 的周线数据")

    # 转换为DataFrame
    result = pd.DataFrame(data_list, columns=rs.fields)
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

    print(f"成功获取股票 {symbol} {len(required_data)} 条周线数据")
    return required_data


def get_stock_data_monthly(symbol, months=100):
    """获取月线数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')

    # 处理股票代码格式
    if symbol.startswith(('sh', 'sz')):
        code = symbol
    elif symbol.startswith('6'):
        code = f"sh.{symbol}"
    elif symbol.startswith(('0', '3')):
        code = f"sz.{symbol}"
    else:
        code = f"sz.{symbol}"

    # 添加随机延迟避免请求过于频繁
    time.sleep(random.uniform(0.5, 1.5))

    # 使用baostock获取月线数据
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="m",
        adjustflag="3"  # 前复权
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        if rs.error_msg:
            raise ValueError(f"获取股票 {symbol} 的月线数据失败: {rs.error_msg}")
        else:
            raise ValueError(f"无法获取股票 {symbol} 的月线数据")

    # 转换为DataFrame
    result = pd.DataFrame(data_list, columns=rs.fields)
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

    print(f"成功获取股票 {symbol} {len(required_data)} 条月线数据")
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


def calculate_ma(data, window):
    """计算移动平均线"""
    return data['close'].rolling(window=window).mean()


def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    exp1 = data['close'].ewm(span=fast).mean()
    exp2 = data['close'].ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def plot_candlestick_with_ma_macd_and_prediction_continuous_short(historical_df, prediction_df, stock_symbol,
                                                                  prediction_days=10, candle_width=0.6):
    """
    绘制短期预测带MA20、MA60、MACD和预测的K线图，消除假期导致的时间间隔，使数据连续显示
    参数:
    - candle_width: K线宽度，控制K线之间的距离 (0.1-1.0)
    """
    # 获取股票名称
    stock_name = get_stock_name(stock_symbol)

    # 合并历史和预测数据
    combined_df = pd.concat([historical_df, prediction_df])

    # 计算移动平均线（历史数据部分）
    historical_df_extended = historical_df.copy()
    historical_df_extended['ma20'] = calculate_ma(historical_df_extended, 20)
    historical_df_extended['ma60'] = calculate_ma(historical_df_extended, 60)

    # 计算预测部分的移动平均线（使用预测的收盘价）
    prediction_df_extended = prediction_df.copy()
    # 对预测数据使用滚动平均，但仅对已有数据进行计算
    all_data = pd.concat([historical_df_extended, prediction_df_extended])
    all_data['ma20'] = calculate_ma(all_data, 20)
    all_data['ma60'] = calculate_ma(all_data, 60)

    # 计算MACD（历史数据部分）
    macd_line, signal_line, histogram = calculate_macd(all_data)

    # 为了消除时间间隔，创建连续的索引
    combined_df_reset = all_data.reset_index()
    combined_df_reset['continuous_index'] = range(len(combined_df_reset))

    # 准备OHLC数据用于绘制K线图（使用连续索引）
    ohlc_list = []
    for i in range(len(combined_df_reset)):
        row = combined_df_reset.iloc[i]
        ohlc_list.append([row['continuous_index'], row['open'], row['high'], row['low'], row['close']])

    # 创建子图布局：上半部分是K线图，中间是成交量图，下面是MACD图
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.01)
    ax1 = plt.subplot(gs[0])  # K线图
    ax2 = plt.subplot(gs[1], sharex=ax1)  # 成交量图
    ax3 = plt.subplot(gs[2], sharex=ax1)  # MACD图

    # 绘制K线图（使用连续索引和可调节的宽度）
    candlestick_ohlc(ax1, ohlc_list, width=candle_width, colorup='red', colordown='green', alpha=0.8)

    # 绘制MA20和MA60线
    ma20_values = combined_df_reset['ma20']
    ma60_values = combined_df_reset['ma60']
    indices = combined_df_reset['continuous_index']

    ax1.plot(indices, ma20_values, color='orange', linewidth=1, label='MA20')
    ax1.plot(indices, ma60_values, color='purple', linewidth=1, label='MA60')

    # 确定历史数据和预测数据的分割点
    hist_len = len(historical_df)

    # 绘制分割竖直线（区分历史和预测数据）
    split_pos = hist_len - 0.5  # 在最后一个历史数据之后绘制分割线
    ax1.axvline(x=split_pos, color='blue', linestyle='--', linewidth=2,
                label=f'Prediction Start', zorder=10)
    ax2.axvline(x=split_pos, color='blue', linestyle='--', linewidth=2, zorder=10)
    ax3.axvline(x=split_pos, color='blue', linestyle='--', linewidth=2, zorder=10)

    # 添加成交量柱状图（使用连续索引，调整柱状图宽度，使用红绿色区分涨跌）
    volumes = combined_df_reset['volume']
    dates_indices = combined_df_reset['continuous_index']

    # 计算涨跌情况，用于成交量着色
    close_prices = combined_df_reset['close']
    price_changes = close_prices.diff().fillna(0)
    volume_colors = ['red' if change >= 0 else 'green' for change in price_changes]

    # 分别设置历史和预测期间的成交量颜色（基于涨跌）
    hist_volumes = combined_df_reset[combined_df_reset['continuous_index'] < hist_len]['volume']
    hist_indices = combined_df_reset[combined_df_reset['continuous_index'] < hist_len]['continuous_index']
    hist_price_changes = price_changes[combined_df_reset['continuous_index'] < hist_len]
    hist_colors = ['red' if change >= 0 else 'green' for change in hist_price_changes]

    pred_volumes = combined_df_reset[combined_df_reset['continuous_index'] >= hist_len]['volume']
    pred_indices = combined_df_reset[combined_df_reset['continuous_index'] >= hist_len]['continuous_index']
    pred_price_changes = price_changes[combined_df_reset['continuous_index'] >= hist_len]
    pred_colors = ['red' if change >= 0 else 'green' for change in pred_price_changes]

    # 历史成交量（根据涨跌着色）- 使用与K线相同的宽度
    ax2.bar(hist_indices, hist_volumes, width=candle_width, color=hist_colors, alpha=0.6, label='Volume (Historical)')
    # 预测成交量（根据涨跌着色）- 使用与K线相同的宽度
    ax2.bar(pred_indices, pred_volumes, width=candle_width, color=pred_colors, alpha=0.6, label='Volume (Predicted)')

    # 绘制MACD图
    # 绘制MACD柱状图 - 修正颜色逻辑
    histogram_values = histogram
    # 修正颜色逻辑：正值为红色（多头），负值为绿色（空头）
    bars = ax3.bar(indices, histogram_values, width=candle_width,
                   color=['red' if x >= 0 else 'green' for x in histogram_values], alpha=0.6)

    # 绘制MACD线和信号线
    ax3.plot(indices, macd_line, color='blue', linewidth=1, label='MACD')
    ax3.plot(indices, signal_line, color='orange', linewidth=1, label='Signal')

    # 设置标题和标签
    ax1.set_title(
        f'{stock_symbol} - Short-term Prediction K-line Chart with MA & MACD (Continuous Time)\nPrediction Period: {prediction_days} days',
        fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax3.set_ylabel('MACD', fontsize=12)
    ax3.set_xlabel('Trading Days', fontsize=12)

    # 设置x轴标签，显示对应的实际日期
    # 每隔一定数量的点显示一个日期标签
    total_points = len(combined_df_reset)
    step = max(1, total_points // 10)  # 为短期预测调整标签数量
    tick_positions = list(range(0, total_points, step))
    tick_labels = [combined_df_reset.iloc[pos]['index'].strftime('%Y-%m-%d') for pos in tick_positions]

    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=45)

    # 添加图例
    # K线图图例
    from matplotlib import lines as mlines
    historical_patch = mpatches.Patch(color='lightgray', alpha=0.3, label='Historical Data')
    predicted_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Predicted Data')
    split_line = mpatches.Patch(color='blue', label='Prediction Boundary')
    ma20_line = mlines.Line2D([], [], color='orange', linewidth=1, label='MA20')
    ma60_line = mlines.Line2D([], [], color='purple', linewidth=1, label='MA60')

    ax1.legend(handles=[historical_patch, predicted_patch, split_line, ma20_line, ma60_line],
               loc='upper left', fontsize=10)

    # 成交量图图例
    up_bar = mpatches.Patch(color='red', alpha=0.6, label='Up Day Volume')
    down_bar = mpatches.Patch(color='green', alpha=0.6, label='Down Day Volume')
    ax2.legend(handles=[up_bar, down_bar], loc='upper left', fontsize=10)

    # MACD图图例
    macd_line_plot = mlines.Line2D([], [], color='blue', linewidth=1, label='MACD')
    signal_line_plot = mlines.Line2D([], [], color='orange', linewidth=1, label='Signal')
    ax3.legend(handles=[macd_line_plot, signal_line_plot], loc='upper left', fontsize=10)

    # 添加背景色区分历史和预测区域
    hist_start, hist_end = 0, hist_len - 1
    pred_start, pred_end = hist_len, total_points - 1

    # K线图的历史和预测区域背景色
    ax1.axvspan(hist_start, hist_end, alpha=0.1, color='lightgray', zorder=0)
    ax1.axvspan(pred_start, pred_end, alpha=0.1, color='lightblue', zorder=0)

    # 成交量图的历史和预测区域背景色
    ax2.axvspan(hist_start, hist_end, alpha=0.1, color='lightgray', zorder=0)
    ax2.axvspan(pred_start, pred_end, alpha=0.1, color='lightblue', zorder=0)

    # MACD图的历史和预测区域背景色
    ax3.axvspan(hist_start, hist_end, alpha=0.1, color='lightgray', zorder=0)
    ax3.axvspan(pred_start, pred_end, alpha=0.1, color='lightblue', zorder=0)

    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 隐藏上面两个子图的x轴标签，但保留刻度线
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    # 确保最下面子图显示x轴标签
    ax3.tick_params(labelbottom=True)

    plt.tight_layout()
    return fig  # 返回图形对象


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


def generate_full_param_combinations():
    """生成完整的参数组合进行测试"""
    # 扩展参数范围，但限制总数以避免内存问题
    T_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # 稍微减少范围
    top_p_values = [0.8, 0.85, 0.9, 0.95]  # 减少top_p的数量
    sample_count_values = [1, 3]  # 减少sample_count的数量

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


def save_investment_strategy_to_csv(buy_sell_strategy, stock_symbol, stock_name, prediction_days, best_param_key,
                                    filename="investment_strategy.csv"):
    """
    将投资策略信息保存到CSV文件中
    """
    # 保持原文件名，不再生成唯一文件名
    if buy_sell_strategy:
        # 计算持股天数
        buy_date = datetime.strptime(buy_sell_strategy['buy_date'], '%Y-%m-%d')
        sell_date = datetime.strptime(buy_sell_strategy['sell_date'], '%Y-%m-%d')
        holding_period = (sell_date - buy_date).days

        # 创建投资策略数据字典
        strategy_data = {
            'stock_name': [stock_name],  # 新增股票名称
            'stock_symbol': [stock_symbol],
            'buy_date': [buy_sell_strategy['buy_date']],
            'buy_price': [buy_sell_strategy['buy_price']],
            'sell_date': [buy_sell_strategy['sell_date']],
            'sell_price': [buy_sell_strategy['sell_price']],
            'expected_return_percent': [buy_sell_strategy['expected_return'] * 100],
            'confidence_score': [buy_sell_strategy['confidence']],
            'holding_period_days': [holding_period],
            'prediction_days': [prediction_days],
            'best_parameters': [best_param_key],
            'run_timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]  # 添加运行时间戳
        }

        # 创建DataFrame
        strategy_df = pd.DataFrame(strategy_data)

        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.isfile(filename)

        # 保存到CSV文件，追加模式
        strategy_df.to_csv(filename, mode='a', header=not file_exists, index=False)

        print(f"投资策略已保存到 {filename}")
    else:
        print("没有有效的投资策略可供保存")


def find_best_parameters_and_generate_kline(stock_symbol, prediction_days=10, candle_width=0.6):
    """
    找到最佳参数组合并生成K线图
    """
    print(f"开始测试股票 {stock_symbol} 的参数组合并生成K线图...")

    # 1. 获取股票名称
    stock_name = get_stock_name(stock_symbol)
    print(f"股票名称: {stock_name}")

    # 2. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 获取股票数据 - 一次性获取大量数据
    print("正在获取日线数据...")
    try:
        daily_df = get_stock_data(stock_symbol, days=200)  # 获取更多历史数据用于短期预测
        if daily_df.empty:
            print("获取的日线数据为空")
            return

        print(f"获取到 {len(daily_df)} 条日线数据")
        daily_df = daily_df.set_index('timestamps')

    except Exception as e:
        print(f"获取日线数据失败: {e}")
        return

    # 获取周线数据
    print("正在获取周线数据...")
    try:
        weekly_df = get_stock_data_weekly(stock_symbol, weeks=100)
        print(f"获取到 {len(weekly_df)} 条周线数据")
        weekly_df = weekly_df.set_index('timestamps')
    except Exception as e:
        print(f"获取周线数据失败: {e}")
        weekly_df = None

    # 获取月线数据
    print("正在获取月线数据...")
    try:
        monthly_df = get_stock_data_monthly(stock_symbol, months=50)
        print(f"获取到 {len(monthly_df)} 条月线数据")
        monthly_df = monthly_df.set_index('timestamps')
    except Exception as e:
        print(f"获取月线数据失败: {e}")
        monthly_df = None

    # 4. 生成参数组合
    param_combinations = generate_full_param_combinations()
    print(f"总共生成 {len(param_combinations)} 种参数组合")

    # 分割数据为训练和验证部分
    split_point = len(daily_df) - prediction_days
    train_df = daily_df[:split_point]
    actual_future = daily_df[split_point:]

    # 准备并行测试参数
    test_args = [
        (train_df, actual_future, model, tokenizer, params, prediction_days)
        for params in param_combinations
    ]

    print(f"开始测试 {len(param_combinations)} 种参数组合...")

    # 使用较少的线程以避免内存问题
    predictions_dict = {}
    evaluation_results = {}

    # 顺序执行而不是并行执行以避免资源冲突
    for i, test_arg in enumerate(test_args):
        param_key, eval_result, success = single_param_test(test_arg)
        if success:
            predictions_dict[param_key] = eval_result['pred_df']
            evaluation_results[param_key] = eval_result
            print(f"  - {param_key}: RMSE={eval_result['RMSE']:.4f}")
        else:
            print(f"  - 参数组合 {test_arg[4]} 测试失败")

        # 每处理完一批参数后进行垃圾回收
        if (i + 1) % 5 == 0:
            gc.collect()

    # 找出最佳参数组合
    if evaluation_results:
        best_param_key = min(evaluation_results.keys(), key=lambda k: evaluation_results[k]["RMSE"])
        best_result = evaluation_results[best_param_key]

        # 获取最佳预测结果
        best_pred_df = predictions_dict.get(best_param_key)
        if best_pred_df is None:
            print("最佳参数组合没有预测结果")
            return

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
        for i, (key, result) in enumerate(sorted_results):  # 显示所有结果
            print(f"{i + 1}. {key} - RMSE: {result['RMSE']:.4f}, MAE: {result['MAE']:.4f}")

        # 使用最佳参数组合重新进行预测（这次是为了生成K线图）
        print(f"使用最佳参数组合 {best_param_key} 进行最终预测...")
        final_prediction_df = predict_with_params(
            daily_df,  # 使用完整数据进行最终预测
            prediction_days,
            model,
            tokenizer,
            T=best_result['params']["T"],
            top_p=best_result['params']["top_p"],
            sample_count=best_result['params']["sample_count"]
        )

        # 应用A股涨跌幅限制
        last_historical_price = daily_df['close'].iloc[-1]
        final_prediction_df = apply_stock_limit_constraints(final_prediction_df, last_historical_price)
        print(f"A股涨跌幅限制已应用到预测结果")

        # 如果预测的成交量有负值或异常值，将其设置为合理范围内的平均值
        final_prediction_df['volume'] = final_prediction_df['volume'].clip(lower=daily_df['volume'].quantile(0.1),
                                                                           upper=daily_df['volume'].quantile(0.9))

        # 生成K线图并保存 - 使用唯一文件名
        print("正在生成带MA/MACD的连续时间轴短期预测K线图...")
        fig = plot_candlestick_with_ma_macd_and_prediction_continuous_short(
            daily_df, final_prediction_df, stock_symbol, prediction_days, candle_width)

        # 保存图形 - 使用唯一文件名（每次运行仍生成唯一图表文件）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_stock_name = stock_name.replace('*', '').replace('/', '').replace('\\', '').replace(':', '').replace('?',
                                                                                                                   '').replace(
            '"', '').replace('<', '').replace('>', '').replace('|', '')
        chart_filename = f"{stock_symbol}_{clean_stock_name}_prediction_chart_{timestamp}.png".replace(
            ' ', '_')
        fig.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"K线图已保存到 {chart_filename}")
        plt.show()

        # 计算最佳买卖策略和可信度
        buy_sell_strategy = calculate_best_buy_sell_strategy(
            daily_df, final_prediction_df, weekly_df, monthly_df
        )

        # 输出预测摘要
        print("\n=== 短期预测结果摘要 ===")
        print(f"股票代码: {stock_symbol}")
        print(f"股票名称: {stock_name}")
        print(f"预测天数: {prediction_days} 天")
        print(f"预测期间价格范围: {final_prediction_df['close'].min():.2f} - {final_prediction_df['close'].max():.2f}")
        print(
            f"预测期间成交量范围: {int(final_prediction_df['volume'].min()):,} - {int(final_prediction_df['volume'].max()):,}")
        print(f"预测起始日期: {final_prediction_df.index[0].strftime('%Y-%m-%d')}")
        print(f"预测结束日期: {final_prediction_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"使用的最佳参数: {best_param_key}")
        print(f"运行时间戳: {timestamp}")

        # 输出交易策略
        print("\n=== 交易策略 ===")
        if buy_sell_strategy:
            print(f"最佳买入时间: {buy_sell_strategy['buy_date']}")
            print(f"最佳买入价格: {buy_sell_strategy['buy_price']:.2f}")
            print(f"最佳卖出时间: {buy_sell_strategy['sell_date']}")
            print(f"最佳卖出价格: {buy_sell_strategy['sell_price']:.2f}")
            print(f"预计涨幅: {buy_sell_strategy['expected_return'] * 100:.2f}%")
            print(f"可信度: {buy_sell_strategy['confidence']:.3f}")
        else:
            print("未能确定最佳交易策略")

        # 保存投资策略到CSV文件（保存到同一个文件）
        print("\n=== 保存投资策略 ===")
        save_investment_strategy_to_csv(buy_sell_strategy, stock_symbol, stock_name, prediction_days, best_param_key)

        # 清理资源
        del model, tokenizer
        gc.collect()
    else:
        print("没有成功生成任何预测结果")


def calculate_best_buy_sell_strategy(daily_df, prediction_df, weekly_df=None, monthly_df=None):
    """
    计算最佳买入卖出策略
    """
    # 获取历史数据的最高价和最低价作为参考
    historical_high = daily_df['high'].max()
    historical_low = daily_df['low'].min()

    # 计算预测数据
    pred_prices = prediction_df['close'].values
    pred_dates = prediction_df.index

    if len(pred_prices) == 0:
        return None

    # 找到预测期内的最低价（买入点）和最高价（卖出点）
    min_idx = np.argmin(pred_prices)
    max_idx = np.argmax(pred_prices)

    # 确保卖出点在买入点之后
    if max_idx < min_idx:
        # 如果最高价在最低价之前，则寻找最低价之后的最高价
        min_price_after_min = pred_prices[min_idx]
        max_price_after_min = min_price_after_min
        max_idx_after_min = min_idx

        for i in range(min_idx + 1, len(pred_prices)):
            if pred_prices[i] > max_price_after_min:
                max_price_after_min = pred_prices[i]
                max_idx_after_min = i

        max_idx = max_idx_after_min

    # 如果最高价仍然在最低价之前，尝试寻找最低价之前的最高价
    if max_idx < min_idx:
        max_price_before_min = pred_prices[0]
        max_idx_before_min = 0

        for i in range(0, min_idx):
            if pred_prices[i] > max_price_before_min:
                max_price_before_min = pred_prices[i]
                max_idx_before_min = i

        max_idx = max_idx_before_min

    # 获取买入和卖出信息
    buy_date = pred_dates[min_idx]
    buy_price = pred_prices[min_idx]
    sell_date = pred_dates[max_idx]
    sell_price = pred_prices[max_idx]

    # 计算预期收益
    expected_return = (sell_price - buy_price) / buy_price if buy_price != 0 else 0

    # 计算可信度（基于多个时间周期的一致性）
    confidence = calculate_confidence_score(
        daily_df, prediction_df, weekly_df, monthly_df, buy_date, sell_date
    )

    return {
        'buy_date': buy_date.strftime('%Y-%m-%d'),
        'buy_price': buy_price,
        'sell_date': sell_date.strftime('%Y-%m-%d'),
        'sell_price': sell_price,
        'expected_return': expected_return,
        'confidence': confidence
    }


def calculate_confidence_score(daily_df, prediction_df, weekly_df, monthly_df, buy_date, sell_date):
    """
    计算预测可信度分数
    """
    # 基础可信度：1.0
    base_confidence = 1.0

    # 根据预测期内的价格波动幅度调整可信度
    price_range = (prediction_df['close'].max() - prediction_df['close'].min()) / prediction_df['close'].mean()
    if price_range > 0.2:  # 波动过大，可信度降低
        base_confidence *= 0.8
    elif price_range > 0.1:  # 中等波动，轻微降低
        base_confidence *= 0.9

    # 如果提供了周线和月线数据，检查趋势一致性
    if weekly_df is not None:
        # 检查周线趋势
        week_before = weekly_df[weekly_df.index < buy_date].tail(1)
        week_after = weekly_df[weekly_df.index >= sell_date].head(1)

        if not week_before.empty and not week_after.empty:
            weekly_trend = (week_after['close'].iloc[0] - week_before['close'].iloc[0]) / week_before['close'].iloc[0]
            if weekly_trend * (sell_date - buy_date).days / 7 > 0:  # 趋势一致
                base_confidence *= 1.1
            else:  # 趋势不一致
                base_confidence *= 0.9

    if monthly_df is not None:
        # 检查月线趋势
        month_before = monthly_df[monthly_df.index < buy_date].tail(1)
        month_after = monthly_df[monthly_df.index >= sell_date].head(1)

        if not month_before.empty and not month_after.empty:
            monthly_trend = (month_after['close'].iloc[0] - month_before['close'].iloc[0]) / month_before['close'].iloc[
                0]
            if monthly_trend * (sell_date - buy_date).days / 30 > 0:  # 趋势一致
                base_confidence *= 1.1
            else:  # 趋势不一致
                base_confidence *= 0.9

    # 确保可信度在合理范围内
    base_confidence = max(0.1, min(1.0, base_confidence))

    return base_confidence


if __name__ == "__main__":
    # 示例：使用最佳参数组合生成K线图
    # stock_symbol = 'sz.002594'  # 使用baostock格式的股票代码
    stock_symbol = 'sh.000001'  # 使用baostock格式的股票代码
    prediction_days = 15  # 短期预测天数
    candle_width = 0.6  # K线宽度，控制K线之间的距离 (0.1-1.0)

    find_best_parameters_and_generate_kline(stock_symbol, prediction_days, candle_width)

    # 登出baostock
    try:
        bs.logout()
        print("已登出baostock")
    except:
        pass
