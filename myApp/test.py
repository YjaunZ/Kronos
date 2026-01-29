import akshare as ak

stockdata = ak.stock_zh_a_hist(
            symbol='001376',
            period="daily",
            start_date='20250403',
            end_date='20260128',
            adjust="qfq"
        )
print(stockdata)