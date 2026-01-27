"""
用kronos预测后面20天内涨幅排名前5的行业
数据获取: 获取所有行业板块指数数据
预测分析: 使用Kronos模型预测未来20天走势
排名筛选: 选出预计涨幅前5的板块
用了predict_batch方法优化效率
"""
import pandas as pd
import numpy as np
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


def get_all_industry_codes():
    """
    获取所有行业板块代码
    """
    try:
        industry_list = ak.stock_board_industry_name_em()
        return industry_list[['板块代码', '板块名称']]
    except Exception as e:
        print(f"获取行业板块列表失败: {e}")
        return None


def batch_predict_industry_growth(codes, days=20, model=None, tokenizer=None):
    """
    批量预测多个板块未来涨幅
    """
    try:
        # 获取所有板块数据
        df_list = []
        for code in codes:
            df = get_stock_data(code, days=200)
            df = df.set_index('timestamps')
            df_list.append(df)

        # 批量预测
        pred_df_list = predict_with_batch_params(
            df_list,
            days,
            model,
            tokenizer,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )

        # 计算涨幅
        results = []
        for i, (df, pred_df) in enumerate(zip(df_list, pred_df_list)):
            current_price = df['close'].iloc[-1]
            future_price = pred_df['close'].iloc[-1]
            growth_rate = (future_price - current_price) / current_price

            results.append({
                'code': codes[i],
                'growth_rate': growth_rate
            })

        return results
    except Exception as e:
        print(f"批量预测失败: {e}")
        return []


def analyze_top_growing_sectors(prediction_days=20, batch_size=50):
    """
    分析涨幅前5的行业板块（使用批量预测）
    """
    print(f"开始分析所有行业板块未来{prediction_days}天的预测涨幅...")

    # 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 获取所有板块
    sectors = get_all_industry_codes()
    if sectors is None:
        print("无法获取行业板块列表")
        return

    print(f"共获取到 {len(sectors)} 个行业板块")

    # 按批次处理，避免内存溢出
    all_results = []
    codes = sectors['板块代码'].tolist()

    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i:i + batch_size]
        print(f"正在预测第{i + 1}-{min(i + batch_size, len(codes))}个板块...")

        batch_results = batch_predict_industry_growth(batch_codes, prediction_days, model, tokenizer)

        # 添加板块名称
        for j, result in enumerate(batch_results):
            code = result['code']
            sector_name = sectors[sectors['板块代码'] == code]['板块名称'].iloc[0]
            result['name'] = sector_name
            all_results.append(result)

    if not all_results:
        print("没有成功预测任何板块")
        return

    # 排名前5
    top_5 = sorted(all_results, key=lambda x: x['growth_rate'], reverse=True)[:5]

    print("\n=== 预计未来20天涨幅前5的行业板块 ===")
    for i, item in enumerate(top_5, 1):
        print(f"{i}. {item['name']} ({item['code']}) - 预计涨幅: {item['growth_rate'] * 100:.2f}%")

    return top_5


def predict_top_growing_sectors():
    """
    预测并返回涨幅前5的行业板块
    """
    return analyze_top_growing_sectors(prediction_days=20)


if __name__ == "__main__":
    # 执行预测
    top_sectors = predict_top_growing_sectors()

    if top_sectors:
        print("\n预测完成！以下是涨幅前5的行业板块：")
        for i, sector in enumerate(top_sectors, 1):
            print(f"{i}. {sector['name']} - 预计涨幅: {sector['growth_rate'] * 100:.2f}%")
    else:
        print("预测失败，没有获得任何结果")
