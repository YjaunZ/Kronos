import pandas as pd
import os
from datetime import datetime
import re

# 从 pridict_stock_price.py 导入必要的函数
from pridict_stock_price import (
    get_stock_data,
    predict_with_params,
    apply_stock_limit_constraints,
    plot_candlestick_with_ma_macd_and_prediction_continuous_short,
    calculate_best_buy_sell_strategy,
    calculate_confidence_score,
    get_stock_name,
    generate_future_trading_days
)

from model import Kronos, KronosTokenizer, KronosPredictor

# 全局变量：存储所有股票的预测数据
all_predictions = []


def format_stock_code(stock_code):
    """
    格式化股票代码为标准格式
    """
    # 将股票代码转换为字符串格式
    code_str = str(stock_code).strip()

    # 如果包含交易所前缀，直接返回
    if code_str.startswith(('sh.', 'sz.')):
        return code_str

    # 检查是否为纯数字，如果是则补零至6位
    if code_str.isdigit():
        # 补零至6位
        code_str = code_str.zfill(6)

    # 根据股票代码前缀确定交易所
    if code_str.startswith('6'):
        formatted_code = f"sh.{code_str}"
    elif code_str.startswith(('0', '3')):
        formatted_code = f"sz.{code_str}"
    else:
        # 如果是6位数字但不是以6、0、3开头，根据长度判断
        if len(code_str) == 6:
            if code_str.startswith('6'):
                formatted_code = f"sh.{code_str}"
            elif code_str.startswith(('0', '3')):
                formatted_code = f"sz.{code_str}"
            else:
                # 默认深圳市场
                formatted_code = f"sz.{code_str}"
        else:
            # 不是6位数字，可能是格式错误，尝试补零
            code_str = code_str.zfill(6)
            if code_str.startswith('6'):
                formatted_code = f"sh.{code_str}"
            else:
                formatted_code = f"sz.{code_str}"

    return formatted_code


def process_stock_codes_from_excel(file_path, prediction_days=10, candle_width=0.6):
    global all_predictions  # 引用全局变量

    # 读取Excel文件
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"文件包含 {len(df)} 行数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 检查是否有'代码'列
    if '代码' not in df.columns:
        print("错误: Excel文件中未找到'代码'列")
        return

    # 提取股票代码并进行格式化
    raw_codes = df['代码'].dropna().unique()
    stock_codes = []
    for code in raw_codes:
        formatted_code = format_stock_code(code)
        stock_codes.append(formatted_code)

    print(f"提取到 {len(stock_codes)} 个唯一股票代码")

    # 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 获取输出目录（与输入文件同目录）
    output_dir = os.path.dirname(file_path)

    # 获取基础文件名（不含扩展名）
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # 清理文件名中的特殊字符
    clean_base_filename = re.sub(r'[^\w\s-]', '_', base_filename)

    print(f"将生成 {len(stock_codes)} 个股票的预测数据和图表...")

    # 为每个股票生成预测
    for i, stock_code in enumerate(stock_codes):
        print(f"\n处理第 {i + 1}/{len(stock_codes)} 个股票: {stock_code}")

        try:
            # 获取股票名称
            stock_name = get_stock_name(stock_code)
            print(f"股票名称: {stock_name}")

            # 获取股票数据
            print("正在获取股票数据...")
            daily_df = get_stock_data(stock_code, days=200)
            if daily_df.empty:
                print(f"获取股票 {stock_code} 数据失败，跳过")
                continue

            daily_df = daily_df.set_index('timestamps')

            # 使用默认参数进行预测
            print("正在进行预测...")
            final_prediction_df = predict_with_params(
                daily_df,
                prediction_days,
                model,
                tokenizer,
                T=1.0,
                top_p=0.9,
                sample_count=1
            )

            # 应用A股涨跌幅限制
            last_historical_price = daily_df['close'].iloc[-1]
            final_prediction_df = apply_stock_limit_constraints(final_prediction_df, last_historical_price)

            # 计算最佳买卖策略
            buy_sell_strategy = calculate_best_buy_sell_strategy(daily_df, final_prediction_df)

            # 构造当前股票的预测数据
            prediction_data = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'prediction_start_date': daily_df.index[-1].strftime('%Y-%m-%d'),
                'prediction_end_date': final_prediction_df.index[-1].strftime('%Y-%m-%d'),
                'current_price': daily_df['close'].iloc[-1],
                'predicted_final_price': final_prediction_df['close'].iloc[-1],
                'predicted_growth_rate': (final_prediction_df['close'].iloc[-1] - daily_df['close'].iloc[-1]) / daily_df['close'].iloc[-1] * 100,
                'prediction_days': prediction_days,
                'best_buy_date': buy_sell_strategy['buy_date'] if buy_sell_strategy else 'N/A',
                'best_buy_price': buy_sell_strategy['buy_price'] if buy_sell_strategy else 'N/A',
                'best_sell_date': buy_sell_strategy['sell_date'] if buy_sell_strategy else 'N/A',
                'best_sell_price': buy_sell_strategy['sell_price'] if buy_sell_strategy else 'N/A',
                'expected_return_percent': buy_sell_strategy['expected_return'] * 100 if buy_sell_strategy else 'N/A',
                'confidence_score': buy_sell_strategy['confidence'] if buy_sell_strategy else 'N/A'
            }

            # 将当前股票的预测数据添加到全局列表
            all_predictions.append(prediction_data)

            # 输出预测摘要
            print(f"股票 {stock_code} ({stock_name}) 预测完成")
            print(f"  - 预测涨幅: {prediction_data['predicted_growth_rate']:.2f}%")

        except Exception as e:
            print(f"处理股票 {stock_code} 时出错: {e}")
            continue

    # 所有股票处理完成后，统一保存到 CSV 文件
    if all_predictions:
        # 转换为 DataFrame
        predictions_df = pd.DataFrame(all_predictions)

        # 生成 CSV 文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{clean_base_filename}_all_predictions_{timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

        # 保存到 CSV 文件
        predictions_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        print(f"\n所有股票预测完成! 共处理 {len(all_predictions)} 个股票")
        print(f"汇总预测数据已保存到: {csv_filepath}")
    else:
        print("没有有效的预测数据可保存")


def main():
    # 指定Excel文件路径 - 修改为上级目录下的data文件夹
    excel_file_path = os.path.join("..", "data", "(2026-02-03 222526).xlsx")

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        print(f"错误: 文件 {excel_file_path} 不存在")
        # 如果相对路径不存在，尝试绝对路径
        excel_file_path = input("请输入Excel文件的完整路径: ")
        if not os.path.exists(excel_file_path):
            print("指定的文件路径无效")
            return

    # 执行处理
    process_stock_codes_from_excel(excel_file_path, prediction_days=10, candle_width=0.6)


if __name__ == "__main__":
    main()
