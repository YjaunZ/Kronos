
"""
用kronos预测后面10天内涨幅排名前5的行业
数据获取: 获取所有行业板块指数数据
预测分析: 使用Kronos模型预测未来10天走势
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
from scipy import stats
import time
import json
from pathlib import Path

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


def get_stock_data_cached(symbol, days=500, cache_dir="./cache"):
    """
    获取指定天数的股票或板块指数数据，带缓存机制
    """
    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # 生成缓存文件名
    cache_file = cache_path / f"{symbol}_{days}_days.json"

    # 检查缓存是否存在且未过期（24小时）
    if cache_file.exists():
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - cache_time).total_seconds() < 24 * 3600:  # 24小时内有效
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                # 转换回DataFrame
                df = pd.DataFrame(cached_data)
                df['timestamps'] = pd.to_datetime(df['timestamps'])
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                print(f"从缓存加载 {symbol} 数据")
                return df
            except Exception as e:
                print(f"读取缓存失败: {e}")

    # 缓存不存在或已过期，重新获取数据
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

    # 保存到缓存
    try:
        cache_dict = required_data.copy()
        cache_dict['timestamps'] = cache_dict['timestamps'].dt.strftime('%Y-%m-%d')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_dict.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存缓存失败: {e}")

    return required_data


def rate_limit_decorator(calls_per_second=1):
    """
    装饰器：限制函数调用频率
    """

    def decorator(func):
        last_called = [0.0]

        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 1.0 / calls_per_second - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret

        return wrapper

    return decorator


@rate_limit_decorator(calls_per_second=0.5)  # 每秒最多0.5次调用，即每2秒一次
def get_single_stock_data(symbol, days=500):
    """带频率限制的单个股票数据获取函数"""
    return get_stock_data_cached(symbol, days)


def get_stock_data(symbol, days=500):
    """获取指定天数的股票或板块指数数据（带缓存和频率限制）"""
    return get_single_stock_data(symbol, days)


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


def predict_with_ensemble_and_confidence(df_list, pred_len, model, tokenizer,
                                         T_values=[0.7, 1.0, 1.3],
                                         top_p_values=[0.8, 0.9, 0.95],
                                         sample_count=3):
    """使用集成方法和置信度评估进行预测，提高准确性"""
    all_pred_results = []

    # 尝试不同的参数组合
    for T in T_values:
        for top_p in top_p_values:
            pred_df_list = predict_with_batch_params(
                df_list,
                pred_len,
                model,
                tokenizer,
                T=T,
                top_p=top_p,
                sample_count=sample_count
            )
            all_pred_results.append(pred_df_list)

    # 对每个板块计算预测值的统计信息
    final_pred_list = []
    confidence_scores = []

    for sector_idx in range(len(df_list)):
        # 收集所有参数组合下的预测结果
        sector_predictions = []
        for pred_result in all_pred_results:
            sector_predictions.append(pred_result[sector_idx])

        # 计算多个预测结果的平均值和置信区间
        close_prices = [pred['close'].iloc[-1] for pred in sector_predictions]
        mean_close = np.mean(close_prices)
        std_close = np.std(close_prices)

        # 使用原始数据的最后价格作为基准
        base_price = df_list[sector_idx]['close'].iloc[-1]

        # 修复：创建新的DataFrame而不是修改副本
        final_pred = sector_predictions[0].copy()

        # 使用 .loc 方法修改最后一行的收盘价
        final_pred.loc[final_pred.index[-1], 'close'] = mean_close

        final_pred_list.append(final_pred)

        # 计算置信分数（标准差越小，置信度越高）
        confidence = 1 / (1 + std_close / base_price)  # 归一化到0-1
        confidence_scores.append(confidence)

    return final_pred_list, confidence_scores


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


def batch_predict_industry_growth_optimized(codes, names, days=10, model=None, tokenizer=None,
                                            use_ensemble=True, confidence_threshold=0.6, batch_size=10):
    """
    优化的批量预测多个板块未来涨幅，支持集成预测和置信度评估
    """
    try:
        # 获取所有板块数据
        df_list = []
        valid_indices = []

        for i, code in enumerate(codes):
            try:
                df = get_stock_data(code, days=200)
                if len(df) > 50:  # 确保有足够的历史数据
                    df = df.set_index('timestamps')
                    df_list.append(df)
                    valid_indices.append(i)

                    # 在每次请求之间加入延迟，避免过于频繁
                    time.sleep(0.5)
            except Exception as e:
                print(f"获取板块 {code} 数据失败: {e}")

        if not df_list:
            print("没有有效的板块数据可供预测")
            return []

        # 根据是否使用集成方法选择预测策略
        if use_ensemble:
            pred_df_list, confidence_scores = predict_with_ensemble_and_confidence(
                df_list, days, model, tokenizer
            )
        else:
            pred_df_list = predict_with_batch_params(
                df_list, days, model, tokenizer, T=1.0, top_p=0.9, sample_count=1
            )
            confidence_scores = [1.0] * len(df_list)  # 如果不使用集成，给默认置信度

        # 计算涨幅和其他指标
        results = []
        for i, (df, pred_df, conf_score) in enumerate(zip(df_list, pred_df_list, confidence_scores)):
            current_price = df['close'].iloc[-1]
            future_price = pred_df['close'].iloc[-1]
            growth_rate = (future_price - current_price) / current_price

            # 只有当置信度高于阈值时才包含在结果中
            if conf_score >= confidence_threshold:
                results.append({
                    'code': codes[valid_indices[i]],
                    'name': names[valid_indices[i]],
                    'growth_rate': growth_rate,
                    'current_price': current_price,
                    'predicted_price': future_price,
                    'confidence_score': conf_score,
                    'volatility': df['close'].pct_change().std(),  # 历史波动率
                    'volume_trend': df['volume'].tail(10).mean() / df['volume'].head(10).mean()  # 成交量趋势
                })

        return results
    except Exception as e:
        print(f"批量预测失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_all_sector_predictions_to_csv(results, filename="all_sector_predictions.csv"):
    """
    将所有行业板块预测结果保存到CSV文件
    """
    import csv
    from datetime import datetime

    # 按涨幅排序
    sorted_results = sorted(results, key=lambda x: x['growth_rate'], reverse=True)

    # 添加排名
    for i, result in enumerate(sorted_results, 1):
        result['rank'] = i
        result['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 写入CSV
    fieldnames = ['rank', 'code', 'name', 'growth_rate', 'current_price', 'predicted_price',
                  'confidence_score', 'volatility', 'volume_trend', 'prediction_date']
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow({
                'rank': result.get('rank', ''),
                'code': result.get('code', ''),
                'name': result.get('name', ''),
                'growth_rate': f"{result.get('growth_rate', 0) * 100:.4f}%",  # 转换为百分比
                'current_price': round(result.get('current_price', 0), 4),
                'predicted_price': round(result.get('predicted_price', 0), 4),
                'confidence_score': round(result.get('confidence_score', 0), 4),
                'volatility': round(result.get('volatility', 0), 6),
                'volume_trend': round(result.get('volume_trend', 0), 4),
                'prediction_date': result.get('prediction_date', '')
            })

    print(f"所有行业板块预测结果已保存到 {filename}，共 {len(sorted_results)} 个板块")


def analyze_all_sectors_with_optimization(prediction_days=10, batch_size=10,
                                          output_filename="all_sector_predictions.csv"):
    """
    分析所有行业板块并保存完整排序结果（使用优化算法）
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

    # 提取代码和名称
    codes = sectors['板块代码'].tolist()
    names = sectors['板块名称'].tolist()

    # 分批处理，降低请求频率
    all_results = []

    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i:i + batch_size]
        batch_names = names[i:i + batch_size]
        print(f"正在预测第{i + 1}-{min(i + batch_size, len(codes))}个板块...")

        batch_results = batch_predict_industry_growth_optimized(
            batch_codes, batch_names, prediction_days, model, tokenizer,
            batch_size=batch_size
        )
        all_results.extend(batch_results)

        # 每处理完一个批次后休息一段时间
        if i + batch_size < len(codes):  # 不是最后一个批次
            print(f"批次处理完成，暂停5秒...")
            time.sleep(5)

    if not all_results:
        print("没有成功预测任何板块")
        return

    # 保存完整结果到CSV
    save_all_sector_predictions_to_csv(all_results, output_filename)

    # 显示前10名
    top_10 = sorted(all_results, key=lambda x: x['growth_rate'], reverse=True)[:10]
    print("\n=== 预计未来10天涨幅前10的行业板块 ===")
    for i, item in enumerate(top_10, 1):
        print(f"{i}. {item['name']} ({item['code']}) - 预计涨幅: {item['growth_rate'] * 100:.2f}% "
              f"(置信度: {item['confidence_score']:.3f})")

    # 显示后10名（跌幅最大）
    bottom_10 = sorted(all_results, key=lambda x: x['growth_rate'])[:10]
    print("\n=== 预计未来10天跌幅前10的行业板块 ===")
    for i, item in enumerate(bottom_10, 1):
        print(f"{i}. {item['name']} ({item['code']}) - 预计跌幅: {item['growth_rate'] * 100:.2f}% "
              f"(置信度: {item['confidence_score']:.3f})")

    return all_results


def predict_top_growing_sectors():
    """
    预测并返回涨幅前5的行业板块
    """
    # 使用优化后的分析函数
    all_results = analyze_all_sectors_with_optimization(
        prediction_days=10,
        batch_size=10,  # 减少批次大小以降低请求频率
        output_filename="industry_predictions_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    )

    if all_results:
        # 获取前5名
        top_5 = sorted(all_results, key=lambda x: x['growth_rate'], reverse=True)[:5]
        return top_5
    else:
        return []


if __name__ == "__main__":
    # 执行优化后的预测
    all_results = analyze_all_sectors_with_optimization(
        prediction_days=10,
        batch_size=10,  # 控制批次大小，减少请求频率
        output_filename="industry_predictions_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    )

    if all_results:
        # 获取前5名
        top_5 = sorted(all_results, key=lambda x: x['growth_rate'], reverse=True)[:5]
        print("\n预测完成！以下是涨幅前5的行业板块：")
        for i, sector in enumerate(top_5, 1):
            print(f"{i}. {sector['name']} ({sector['code']}) - 预计涨幅: {sector['growth_rate'] * 100:.2f}% "
                  f"(置信度: {sector['confidence_score']:.3f})")
    else:
        print("预测失败，没有获得任何结果")
