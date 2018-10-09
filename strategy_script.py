"""
策略思路：
- 震荡市中的反转策略
- 某段时间价格变化幅度和过去某段时间的价格变化幅度有一定相关性
- 可以使用历史价格数据预测未来某段时间价格变化幅度
- 如果价格低破一定水平则会反弹

交易流程：
- 每四小时作为一个区间
- 在每四小时末：
    - 计算过去48小时和72小时的range，记为R和R2
- 在下一段四小时初：
    - 已知此四小时open价
    - 挂一个限价单，long_limit(t) = open(t) - 0.12 * R(t-1)
    - 如果交易成功，就设置50点止损，R*0.32作为盈利目标，但买入时间段不止盈
    - 如果此四小时没有交易成功，则撤销限价单
    - 总条件：R2要在区间，否则什么都不做
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
************** 读取数据 ******************************************************
"""
csv_file_name = 'EURUSD240.csv'

data = pd.DataFrame(pd.read_csv(csv_file_name, header=None))
# print(data)

data.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
# print(data)

"""
************** 数据处理 ******************************************************
"""
data['pct_change'] = data['close'] / data['open'] - 1

data['high_48h'] = data['high'].rolling(12).max()
data['low_48h'] = data['low'].rolling(12).min()

data['high_72h'] = data['high'].rolling(18).max()
data['low_72h'] = data['low'].rolling(18).min()

data['R'] = data['high_48h'] - data['low_48h']
data['R2'] = data['high_72h'] - data['low_72h']
# print(data)

data['long_limit'] = data['open'] - 0.12 * data['R'].shift(1)
# print(data)

"""
************** 交易流程 ******************************************************
"""
position = 0

for i in range(18, len(data)):

    # 无仓位进行开仓
    if position == 0:
        if 0.013 <= data.loc[i-1, 'R2'] <= 0.019:

            # 如果最低价低于限价单价格，触及long_limit则买入
            if data.loc[i, 'low'] <= data.loc[i, 'long_limit']:
                position = 1
                long_open_price = data.loc[i, 'long_limit']
                stop_win_price = long_open_price + 0.32 * data.loc[i-1, 'R']
                stop_lose_price = long_open_price - 0.0050

                # 如果最低价低于止损价格，则说明买入后价格继续下跌，以止损价卖出
                if data.loc[i, 'low'] < stop_lose_price:
                    position = 0
                    # 收益使用HPR
                    data.loc[i, 'return'] = stop_lose_price/long_open_price - 1
                # 持有到四小时末
                else:
                    data.loc[i, 'return'] = data.loc[i, 'close'] / long_open_price - 1

    # 有仓位进行平仓
    else:

        # 如果最低价低于止损价格，则平仓
        if data.loc[i, 'low'] < stop_lose_price:
            position = 0
            data.loc[i, 'return'] = min(stop_lose_price, data.loc[i, 'open']) / data.loc[i, 'open'] - 1
        elif data.loc[i, 'high'] > stop_win_price:
            position = 0
            data.loc[i, 'return'] = max(stop_win_price, data.loc[i, 'open']) / data.loc[i, 'open'] - 1
        else:
            data.loc[i, 'return'] = data.loc[i, 'close'] / data.loc[i, 'open'] - 1

"""
************** 计算收益 ******************************************************
"""
data['return'].fillna(0, inplace=True)
data['strategy_return'] = (data['return'] + 1).cumprod()
data['forex_return'] = (data['pct_change'] + 1).cumprod()

matplotlib.style.use('ggplot')
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(data.forex_return)
ax.plot(data.strategy_return)
plt.legend()
plt.show()
