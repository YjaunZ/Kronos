import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
import akshare as ak

def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 获取特定股票数据
stocksymbol = '002366'
period = 'daily'
start_time = '20240101'
# end_time = '20260723'
# 使用最近的实际交易数据作为基础
end_time = pd.Timestamp.now().strftime('%Y%m%d')  # 当前日期

data_path = './data/'+stocksymbol+'.csv'
stockdata = ak.stock_zh_a_hist(stocksymbol, period, start_time, end_time, adjust="qfq")
# 提取所需字段并重命名为模型需要的格式
required_data = stockdata[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']].copy()
# 重命名列以符合模型要求
column_mapping = {
    '日期': 'timestamps',
    '开盘': 'open',
    '最高': 'high',
    '最低': 'low',
    '收盘': 'close',
    '成交量': 'volume',
    '成交额': 'amount'
}
required_data.rename(columns=column_mapping, inplace=True)
# 转换时间戳格式
required_data['timestamps'] = pd.to_datetime(required_data['timestamps'])
# 保存到CSV文件
required_data.to_csv(data_path, index=False)


# 3. Prepare Data
df = pd.read_csv(data_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define context window and prediction length

lookback = 400
pred_len = 100

# Prepare inputs for the predictor
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
# A pandas Series of timestamps corresponding to the historical data in df.
x_timestamp = df.loc[:lookback-1, 'timestamps']
#A pandas Series of timestamps for the future periods you want to predict.
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']



# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,              # Temperature for sampling
    top_p=0.9,          # Nucleus sampling probability
    sample_count=1,     # Number of forecast paths to generate and average
    verbose=True
)
# 设置预测数据的时间戳索引
pred_df.index = y_timestamp

# # 5. Visualize Results
# print("Forecasted Data Head:")
# print(pred_df.head())
# 保存预测数据
prediction_output_path = f'./data/prediction_{stocksymbol}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
pred_df.to_csv(prediction_output_path)
print(f"预测数据已保存到: {prediction_output_path}")

# 创建包含历史数据和预测数据的完整表格
historical_data = x_df.copy()
historical_data.index = x_timestamp

# 合并历史数据和预测数据
full_data = pd.concat([historical_data, pred_df])

# 保存完整数据
full_output_path = f'./data/full_data_with_prediction_{stocksymbol}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
full_data.to_csv(full_output_path)
print(f"完整数据已保存到: {full_output_path}")

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1].copy()
kline_df['timestamps'] = pd.to_datetime(kline_df['timestamps'])
kline_df.set_index('timestamps', inplace=True)

# visualize
plot_prediction(kline_df, pred_df)

