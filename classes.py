import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class Data:
    """
    Each instance represents a pandas dataframe
    """

    def __init__(self, csv_file_name, k1, k2):
        self.df = pd.DataFrame(pd.read_csv(csv_file_name, header=None))
        self.k1_long_limit = k1
        self.k2_stop_win = k2
        self.return_ = None
        self.std = None
        self.sharpe_ratio = None

    def init_columns(self):
        self.df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

    def add_pct_change(self):
        self.df['pct_change'] = self.df['close'] / self.df['open'] - 1

    def add_ranges(self):
        self.df['high_48h'] = self.df['high'].rolling(12).max()
        self.df['low_48h'] = self.df['low'].rolling(12).min()

        self.df['high_72h'] = self.df['high'].rolling(18).max()
        self.df['low_72h'] = self.df['low'].rolling(18).min()

        self.df['R'] = self.df['high_48h'] - self.df['low_48h']
        self.df['R2'] = self.df['high_72h'] - self.df['low_72h']

    def add_long_limit(self):
        self.df['long_limit'] = self.df['open'] - self.k1_long_limit * self.df['R'].shift(1)

    def df_process(self):
        self.init_columns()
        self.add_pct_change()
        self.add_ranges()
        self.add_long_limit()

    def strategy(self):
        self.df_process()

        position = 0
        for i in range(18, len(self.df)):

            # 无仓位进行开仓
            if position == 0:
                if 0.013 <= self.df.loc[i - 1, 'R2'] <= 0.019:

                    # 如果最低价低于限价单价格，触及long_limit则买入
                    if self.df.loc[i, 'low'] <= self.df.loc[i, 'long_limit']:
                        position = 1
                        long_open_price = self.df.loc[i, 'long_limit']
                        stop_win_price = long_open_price + self.k2_stop_win * self.df.loc[i - 1, 'R']
                        stop_lose_price = long_open_price - 0.0050

                        # 如果最低价低于止损价格，则说明买入后价格继续下跌，以止损价卖出
                        if self.df.loc[i, 'low'] < stop_lose_price:
                            position = 0
                            # 收益使用HPR
                            self.df.loc[i, 'return'] = stop_lose_price / long_open_price - 1
                        # 持有到四小时末
                        else:
                            self.df.loc[i, 'return'] = self.df.loc[i, 'close'] / long_open_price - 1

            # 有仓位进行平仓
            else:

                # 如果最低价低于止损价格，则平仓
                if self.df.loc[i, 'low'] < stop_lose_price:
                    position = 0
                    self.df.loc[i, 'return'] = min(stop_lose_price, self.df.loc[i, 'open']) / self.df.loc[i, 'open'] - 1
                elif self.df.loc[i, 'high'] > stop_win_price:
                    position = 0
                    self.df.loc[i, 'return'] = max(stop_win_price, self.df.loc[i, 'open']) / self.df.loc[i, 'open'] - 1
                else:
                    self.df.loc[i, 'return'] = self.df.loc[i, 'close'] / self.df.loc[i, 'open'] - 1

        self.df['return'].fillna(0, inplace=True)

    def plot_strategy(self):
        self.strategy()
        self.df['strategy_return'] = (self.df['return'] + 1).cumprod()
        self.df['forex_return'] = (self.df['pct_change'] + 1).cumprod()

        matplotlib.style.use('ggplot')
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.df.forex_return)
        ax.plot(self.df.strategy_return)
        plt.legend()
        plt.show()

    def cpt_r_std_SR(self):
        self.strategy()
        self.return_ = self.df['return'].mean()
        self.std = self.df['return'].std()
        self.sharpe_ratio = self.return_ / self.std
