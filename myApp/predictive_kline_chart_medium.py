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
    """获取指定天数的股票数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

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


def predict_stock_for_duration(df, pred_len, model, tokenizer):
    """对指定长度进行预测"""
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
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False
    )

    # 设置预测数据的时间戳索引
    pred_df.index = y_timestamp

    return pred_df


def plot_candlestick_with_ma_macd_and_prediction_continuous_medium(historical_df, prediction_df, stock_symbol,
                                                                 prediction_days=30, candle_width=0.6):
    """
    绘制中线预测带MA20、MA60、MACD和预测的K线图，消除假期导致的时间间隔，使数据连续显示
    参数:
    - candle_width: K线宽度，控制K线之间的距离 (0.1-1.0)
    """
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
    fig = plt.figure(figsize=(18, 14))
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

    # 添加成交量柱状图（使用连续索引，调整柱状图宽度）
    volumes = combined_df_reset['volume']
    dates_indices = combined_df_reset['continuous_index']

    # 分别设置历史和预测期间的成交量颜色
    hist_volumes = combined_df_reset[combined_df_reset['continuous_index'] < hist_len]['volume']
    hist_indices = combined_df_reset[combined_df_reset['continuous_index'] < hist_len]['continuous_index']
    pred_volumes = combined_df_reset[combined_df_reset['continuous_index'] >= hist_len]['volume']
    pred_indices = combined_df_reset[combined_df_reset['continuous_index'] >= hist_len]['continuous_index']

    # 历史成交量（灰色）- 使用与K线相同的宽度
    ax2.bar(hist_indices, hist_volumes, width=candle_width, color='lightgray', alpha=0.6, label='Volume (Historical)')
    # 预测成交量（浅蓝色）- 使用与K线相同的宽度
    ax2.bar(pred_indices, pred_volumes, width=candle_width, color='lightblue', alpha=0.6, label='Volume (Predicted)')

    # 绘制MACD图
    # 绘制MACD柱状图
    histogram_values = histogram
    bars = ax3.bar(indices, histogram_values, width=candle_width,
                   color=['green' if x >= 0 else 'red' for x in histogram_values], alpha=0.6)

    # 绘制MACD线和信号线
    ax3.plot(indices, macd_line, color='blue', linewidth=1, label='MACD')
    ax3.plot(indices, signal_line, color='orange', linewidth=1, label='Signal')

    # 设置标题和标签
    ax1.set_title(
        f'{stock_symbol} - Medium-term Prediction K-line Chart with MA & MACD (Continuous Time)\nPrediction Period: {prediction_days} days',
        fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax3.set_ylabel('MACD', fontsize=12)
    ax3.set_xlabel('Trading Days', fontsize=12)

    # 设置x轴标签，显示对应的实际日期
    # 每隔一定数量的点显示一个日期标签
    total_points = len(combined_df_reset)
    step = max(1, total_points // 15)  # 为中线预测增加更多标签
    tick_positions = list(range(0, total_points, step))
    tick_labels = [combined_df_reset.iloc[pos]['index'].strftime('%Y-%m-%d') for pos in tick_positions]

    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=45)

    # 添加图例
    # K线图图例
    historical_patch = mpatches.Patch(color='lightgray', alpha=0.3, label='Historical Data')
    predicted_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Predicted Data')
    split_line = mpatches.Patch(color='blue', label='Prediction Boundary')
    ma20_line = mlines.Line2D([], [], color='orange', linewidth=1, label='MA20')
    ma60_line = mlines.Line2D([], [], color='purple', linewidth=1, label='MA60')

    ax1.legend(handles=[historical_patch, predicted_patch, split_line, ma20_line, ma60_line],
               loc='upper left', fontsize=10)

    # 成交量图图例
    hist_vol_patch = mpatches.Patch(color='lightgray', alpha=0.6, label='Historical Volume')
    pred_vol_patch = mpatches.Patch(color='lightblue', alpha=0.6, label='Predicted Volume')
    ax2.legend(handles=[hist_vol_patch, pred_vol_patch], loc='upper left', fontsize=10)

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
    plt.show()


def generate_medium_term_prediction_kline_with_ma_macd_continuous(stock_symbol, prediction_days=30, candle_width=0.6):
    """
    生成带MA20、MA60、MACD和成交量的中线预测K线图的主要函数（连续时间轴版本）
    参数:
    - candle_width: K线宽度，控制K线之间的距离 (0.1-1.0)
    """
    print(f"开始生成股票 {stock_symbol} 的带MA/MACD的中线预测K线图（连续时间轴）...")
    print(f"K线宽度设置为: {candle_width}")
    print(f"中线预测天数: {prediction_days} 天")

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
        df = get_stock_data(stock_symbol, days=300)  # 获取近300天数据用于中线预测
        if df.empty:
            print("获取的股票数据为空")
            return

        print(f"获取到 {len(df)} 条历史数据")
        df = df.set_index('timestamps')

    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return

    # 3. 执行中线预测
    print(f"正在进行 {prediction_days} 天的中线预测...")
    try:
        prediction_df = predict_stock_for_duration(df, prediction_days, model, tokenizer)
        print(f"中线预测完成，预测了 {len(prediction_df)} 天")

        # 如果预测的成交量有负值或异常值，将其设置为合理范围内的平均值
        prediction_df['volume'] = prediction_df['volume'].clip(lower=df['volume'].quantile(0.1),
                                                               upper=df['volume'].quantile(0.9))
    except Exception as e:
        print(f"预测失败: {e}")
        return

    # 4. 绘制K线图（包含MA20、MA60、MACD、成交量，使用连续时间轴）
    print("正在生成带MA/MACD的连续时间轴中线预测K线图...")
    plot_candlestick_with_ma_macd_and_prediction_continuous_medium(df, prediction_df, stock_symbol, prediction_days,
                                                            candle_width)

    # 5. 输出预测摘要
    print("\n=== 中线预测结果摘要 ===")
    print(f"股票代码: {stock_symbol}")
    print(f"预测天数: {prediction_days} 天")
    print(f"预测期间价格范围: {prediction_df['close'].min():.2f} - {prediction_df['close'].max():.2f}")
    print(f"预测期间成交量范围: {int(prediction_df['volume'].min()):,} - {int(prediction_df['volume'].max()):,}")
    print(f"预测起始日期: {prediction_df.index[0].strftime('%Y-%m-%d')}")
    print(f"预测结束日期: {prediction_df.index[-1].strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    # 导入matplotlib.lines用于图例
    from matplotlib import lines as mlines

    # 示例：生成带MA/MACD的中线预测K线图（连续时间轴）
    stock_symbol = '002131'  # 可以修改为目标股票代码
    prediction_days = 30  # 中线预测天数
    candle_width = 0.6  # K线宽度，控制K线之间的距离 (0.1-1.0)

    generate_medium_term_prediction_kline_with_ma_macd_continuous(stock_symbol, prediction_days, candle_width)
