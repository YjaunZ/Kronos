import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import holidays
import akshare as ak
import matplotlib
import platform
from model import Kronos, KronosTokenizer, KronosPredictor
# 根据操作系统自动选择后端
system = platform.system()
if system == "Darwin":  # macOS
    matplotlib.use('MacOSX')
elif system == "Windows":
    matplotlib.use('TkAgg')
else:  # Linux和其他系统
    matplotlib.use('TkAgg')



def generate_future_trading_days(start_date, num_days):
    """
    生成未来交易日时间戳（排除周末和节假日）

    Parameters:
    start_date: 开始日期，可以是字符串或datetime对象
    num_days: 需要生成的交易日数量

    Returns:
    list: 包含未来交易日的pandas Timestamp列表
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


def plot_prediction_with_future(kline_df, pred_df, stock_symbol):
    """
    绘制历史数据与未来预测数据的对比图
    """
    # 确保 pred_df 使用正确的时间索引
    if not isinstance(pred_df.index, pd.DatetimeIndex):
        print("警告: pred_df 索引不是时间戳格式")
        return

    # 准备数据
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Historical'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Historical'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # 绘制收盘价
    ax1.plot(close_df.index[:-len(sr_pred_close)], close_df['Historical'][:-len(sr_pred_close)],
             label='Historical', color='blue', linewidth=1.5, linestyle='-')
    ax1.plot(close_df.index[-len(sr_pred_close):], close_df['Prediction'][-len(sr_pred_close):],
             label='Prediction', color='red', linewidth=2.0, linestyle='-')
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, linestyle='-', alpha=0.6)
    ax1.set_title(f'{stock_symbol} Stock Price Forecast', fontsize=16, fontweight='bold')

    # 绘制成交量
    ax2.plot(volume_df.index[:-len(sr_pred_volume)], volume_df['Historical'][:-len(sr_pred_volume)],
             label='Historical', color='blue', linewidth=1.5, linestyle='-')
    ax2.plot(volume_df.index[-len(sr_pred_volume):], volume_df['Prediction'][-len(sr_pred_volume):],
             label='Prediction', color='red', linewidth=2.0, linestyle='-')
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, linestyle='-', alpha=0.6)

    # 设置日期格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))  # 每4周显示一个标签
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def predict_stock_future(stock_symbol, pred_days=50):
    """
    预测A股特定股票未来股价的主要函数

    Parameters:
    stock_symbol: 股票代码，如 '002366'
    pred_days: 预测天数，默认50天
    """
    print(f"开始预测股票 {stock_symbol} 未来 {pred_days} 天的股价...")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

    # 2. 获取股票数据
    print("正在获取股票数据...")
    try:
        # 获取最近一年的数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=500)).strftime('%Y%m%d')

        stockdata = ak.stock_zh_a_hist(
            symbol=stock_symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if stockdata.empty:
            print("获取的股票数据为空")
            return None

        print(f"获取到 {len(stockdata)} 条历史数据")
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return None

    # 3. 数据预处理
    print("正在预处理数据...")
    try:
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

        # 检查是否有足够的数据
        if len(required_data) < 100:
            print("历史数据不足，至少需要100天数据")
            return None

    except Exception as e:
        print(f"数据预处理失败: {e}")
        return None

    # 4. 准备预测参数
    print("正在准备预测参数...")
    try:
        df = required_data.set_index('timestamps')

        # 确定回看窗口和预测长度
        max_lookback = min(250, len(df) - 50)
        lookback = max_lookback
        pred_len = min(pred_days, 50)

        print(f"使用回看窗口: {lookback} 天，预测长度: {pred_len} 天")

        # 准备输入数据
        x_df = df.iloc[-lookback:, :][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = df.index[-lookback:]

        # 生成未来交易日时间戳
        future_timestamps = generate_future_trading_days(df.index[-1], pred_len)
        y_timestamp = pd.DatetimeIndex(future_timestamps)

        print(f"历史数据范围: {x_timestamp[0]} 到 {x_timestamp[-1]}")
        print(f"预测时间范围: {y_timestamp[0]} 到 {y_timestamp[-1]}")

    except Exception as e:
        print(f"准备预测参数失败: {e}")
        return None

    # 5. 执行预测 - 使用修改后的参数
    print("正在执行预测...")
    try:
        # 尝试将时间戳转换为pandas Series格式（这是模型期望的格式）
        x_timestamp_series = pd.Series(x_timestamp)
        y_timestamp_series = pd.Series(y_timestamp)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp_series,
            y_timestamp=y_timestamp_series,
            pred_len=pred_len,
            T=1.0,  # 温度参数
            top_p=0.9,  # 核采样概率
            sample_count=1,  # 采样次数
            verbose=True
        )

        # 设置预测数据的时间戳索引
        pred_df.index = y_timestamp

        print("预测完成！")

    except AttributeError as ae:
        if "'DatetimeIndex' object has no attribute 'dt'" in str(ae) or "'numpy.ndarray' object has no attribute 'dt'" in str(ae):
            print("检测到时间戳格式问题，正在尝试备用方法...")
            try:
                # 尝试使用pandas Series格式，但转换为datetime对象列表
                x_timestamp_series = pd.Series(x_timestamp.to_pydatetime())
                y_timestamp_series = pd.Series(y_timestamp.to_pydatetime())

                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp_series,
                    y_timestamp=y_timestamp_series,
                    pred_len=pred_len,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=True
                )

                # 设置预测数据的时间戳索引
                pred_df.index = y_timestamp

                print("预测完成！")
            except Exception as e:
                print(f"备用方法预测也失败: {e}")
                return None
        else:
            print(f"属性错误: {ae}")
            return None
    except Exception as e:
        print(f"预测过程失败: {e}")
        return None

    # 6. 保存预测结果
    print("正在保存预测结果...")
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_output_path = f'./data/prediction_{stock_symbol}_{timestamp_str}.csv'

        # 添加预测标识列
        pred_df['is_prediction'] = True
        x_df['is_prediction'] = False

        # 合并历史数据和预测数据
        full_data = pd.concat([
            x_df.assign(type='historical'),
            pred_df.assign(type='predicted')
        ])

        full_data.to_csv(prediction_output_path)
        print(f"预测数据已保存到: {prediction_output_path}")

        # 保存仅预测数据
        prediction_only_path = f'./data/prediction_only_{stock_symbol}_{timestamp_str}.csv'
        pred_df.to_csv(prediction_only_path)
        print(f"仅预测数据已保存到: {prediction_only_path}")

    except Exception as e:
        print(f"保存预测结果失败: {e}")
        return None

    # 7. 可视化结果
    print("正在生成可视化图表...")
    try:
        plot_prediction_with_future(x_df, pred_df, stock_symbol)
    except Exception as e:
        print(f"可视化失败: {e}")

    # 8. 输出预测摘要
    print("\n=== 预测结果摘要 ===")
    print(f"股票代码: {stock_symbol}")
    print(f"预测期间: {y_timestamp[0].date()} 到 {y_timestamp[-1].date()}")
    print(f"预测天数: {len(pred_df)} 天")
    print(f"预测开盘价范围: {pred_df['open'].min():.2f} - {pred_df['open'].max():.2f}")
    print(f"预测收盘价范围: {pred_df['close'].min():.2f} - {pred_df['close'].max():.2f}")
    print(f"预测最高价范围: {pred_df['high'].min():.2f} - {pred_df['high'].max():.2f}")
    print(f"预测最低价范围: {pred_df['low'].min():.2f} - {pred_df['low'].max():.2f}")

    return pred_df



if __name__ == "__main__":
    # 示例：预测特定股票
    stock_symbol = '601012'  # 可以修改为目标股票代码
    pred_days = 50

    result = predict_stock_future(stock_symbol, pred_days)

    if result is not None:
        print(f"\n成功预测股票 {stock_symbol} 未来 {pred_days} 天的股价")
        print("预测数据已保存至本地文件")
    else:
        print(f"\n预测股票 {stock_symbol} 失败")
