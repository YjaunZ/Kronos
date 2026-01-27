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

import matplotlib.pyplot as plt
'''
1. 短期预测（1-10天）
    训练天数: 60-120天
    预测天数: 1-10天
    适用场景: 日内交易、短期趋势判断
    优势: 模型更容易捕捉短期模式
2. 中期预测（11-30天）
    训练天数: 120-250天
    预测天数: 11-30天
    适用场景: 月度投资规划
    优势: 平衡短期波动和中期趋势
3. 长期预测（31-100天）
    训练天数: 250-400天
    预测天数: 31-100天
    适用场景: 季度或年度投资策略
    注意事项: 长期预测准确性会显著下降
具体建议
1. 基于Kronos模型特性优化
最大上下文限制: 512（由max_context=512决定）
推荐比例: 训练天数:预测天数 = 4:1 至 8:1
示例搭配:
训练300天 → 预测50天（比例6:1）
训练350天 → 预测40天（比例8.75:1）
训练400天 → 预测30天（比例13.3:1）

'''


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


def get_available_industry_codes():
    """
    获取所有可用的行业板块代码
    """
    try:
        industry_list = ak.stock_board_industry_name_em()
        print(f"共找到 {len(industry_list)} 个行业板块")
        print(industry_list[['板块代码', '板块名称']].head(10))
        return industry_list
    except Exception as e:
        print(f"获取行业板块列表失败: {e}")
        return None


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


def plot_comparison_predictions(historical_df, short_pred, medium_pred, long_pred, stock_symbol):
    """
    绘制不同预测时长的对比图
    """
    # 准备数据
    historical_close = historical_df['close']

    fig, ax = plt.subplots(figsize=(16, 10))

    # 绘制历史数据
    ax.plot(historical_close.index, historical_close,
            label='Historical', color='black', linewidth=1.5, linestyle='-', zorder=5)

    # 绘制不同预测结果
    if short_pred is not None and not short_pred.empty:
        ax.plot(short_pred.index, short_pred['close'],
                label=f'Short-term Prediction ({len(short_pred)} days)',
                color='red', linewidth=2, linestyle='-', marker='o', markersize=4, zorder=4)

    if medium_pred is not None and not medium_pred.empty:
        ax.plot(medium_pred.index, medium_pred['close'],
                label=f'Medium-term Prediction ({len(medium_pred)} days)',
                color='orange', linewidth=2, linestyle='-', marker='s', markersize=4, zorder=3)

    if long_pred is not None and not long_pred.empty:
        ax.plot(long_pred.index, long_pred['close'],
                label=f'Long-term Prediction ({len(long_pred)} days)',
                color='blue', linewidth=2, linestyle='-', marker='^', markersize=4, zorder=2)

    ax.set_ylabel('Price', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title(f'{stock_symbol} - Comparison of Different Prediction Horizons', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 设置日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))  # 每6周显示一个标签
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def compare_predictions(stock_symbol, short_days=10, medium_days=30, long_days=60):
    """
    对比不同预测时长的结果
    """
    print(
        f"开始对比{'板块指数' if stock_symbol.startswith('BK') else '股票'} {stock_symbol} 不同预测时长的结果...")

    # 检查是否为板块指数
    if stock_symbol.startswith('BK'):
        print(f"检测到板块指数代码: {stock_symbol}")
        try:
            available_codes = get_available_industry_codes()
        except:
            print("无法获取行业板块列表")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 获取股票或板块指数数据
    print("正在获取数据...")
    try:
        df = get_stock_data(stock_symbol, days=500)
        if df.empty:
            print("获取的数据为空")
            return

        print(f"获取到 {len(df)} 条历史数据")
        df = df.set_index('timestamps')

    except Exception as e:
        print(f"获取数据失败: {e}")
        return

    # 3. 执行不同长度的预测
    print("正在进行不同长度的预测...")

    # 短期预测
    short_pred = None
    try:
        print(f"执行短期预测 ({short_days} 天)...")
        short_pred = predict_stock_for_duration(df, short_days, model, tokenizer)
        print(f"短期预测完成，预测了 {len(short_pred)} 天")
    except Exception as e:
        print(f"短期预测失败: {e}")

    # 中期预测
    medium_pred = None
    try:
        print(f"执行中期预测 ({medium_days} 天)...")
        medium_pred = predict_stock_for_duration(df, medium_days, model, tokenizer)
        print(f"中期预测完成，预测了 {len(medium_pred)} 天")
    except Exception as e:
        print(f"中期预测失败: {e}")

    # 长期预测
    long_pred = None
    try:
        print(f"执行长期预测 ({long_days} 天)...")
        long_pred = predict_stock_for_duration(df, long_days, model, tokenizer)
        print(f"长期预测完成，预测了 {len(long_pred)} 天")
    except Exception as e:
        print(f"长期预测失败: {e}")

    # 4. 绘制对比图
    print("正在生成对比图表...")
    plot_comparison_predictions(df, short_pred, medium_pred, long_pred, stock_symbol)

    # 5. 输出预测摘要
    print("\n=== 预测结果摘要 ===")
    print(f"{'板块指数' if stock_symbol.startswith('BK') else '股票'}代码: {stock_symbol}")

    if short_pred is not None:
        print(
            f"短期预测 ({short_days} 天) - 收盘价范围: {short_pred['close'].min():.2f} - {short_pred['close'].max():.2f}")

    if medium_pred is not None:
        print(
            f"中期预测 ({medium_days} 天) - 收盘价范围: {medium_pred['close'].min():.2f} - {medium_pred['close'].max():.2f}")

    if long_pred is not None:
        print(
            f"长期预测 ({long_days} 天) - 收盘价范围: {long_pred['close'].min():.2f} - {long_pred['close'].max():.2f}")


if __name__ == "__main__":
    # 示例：对比不同预测时长
    stock_symbol = 'BK0427'  # 可以修改为目标股票代码或板块代码如 'BK1033'

    # 定义不同的预测时长
    short_term = 10  # 短线预测
    medium_term = 30  # 中长线预测
    long_term = 60  # 长线预测

    compare_predictions(stock_symbol, short_term, medium_term, long_term)
