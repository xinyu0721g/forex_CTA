import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class Data:
    """
    Each instance represents one pandas DataFrame with different currency pairs, time intervals, time periods, etc.
    """

    def __init__(self, csv_file_name, param1, param2):
        """
        Strategy results change with data set and parameters.
        :param csv_file_name: data set (different currency pairs/time interval)
        :param param1: to determine long limit price
        :param param2: to determine stop win price
        """
        self.df = pd.DataFrame(pd.read_csv(csv_file_name, header=None))
        self.k1_long_limit = param1
        self.k2_stop_win = param2
        self.effect_row = 0  # start row in strategy
        self.R2_low_limit = 0.013
        self.R2_high_limit = 0.019
        self.s_return = None
        self.s_std = None
        self.s_SR = None
        self.f_return = None
        self.f_std = None
        self.f_SR = None

    def init_columns(self):
        """
        add titles for initials columns.
        :return: None
        """
        self.df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

    def add_pct_change(self):
        """
        pct_change is the HPR in one time interval.
        :return: None
        """
        self.df['pct_change'] = self.df['close'] / self.df['open'] - 1

    def add_ranges(self):
        """
        different for different time intervals. Specified in sub classes.
        :return: None
        """
        pass

    def add_long_limit(self):
        """
        set entry condition.
        :return: None
        """
        self.df['long_limit'] = self.df['open'] - self.k1_long_limit * self.df['R'].shift(1)

    def df_process(self):
        self.init_columns()
        self.add_pct_change()
        self.add_ranges()
        self.add_long_limit()

    def strategy(self):
        self.df_process()

        position = 0
        for i in range(self.effect_row, len(self.df)):

            # 无仓位进行开仓
            if position == 0:
                if self.R2_low_limit <= self.df.loc[i - 1, 'R2'] <= self.R2_high_limit:

                    # 如果最低价低于限价单价格，触及long_limit则买入
                    if self.df.loc[i, 'low'] <= self.df.loc[i, 'long_limit']:
                        position = 1
                        long_open_price = self.df.loc[i, 'long_limit']
                        stop_win_price = long_open_price + self.k2_stop_win * self.df.loc[i - 1, 'R']
                        stop_lose_price = long_open_price - 0.0050

                        # 如果最低价低于止损价格，则说明买入后价格继续下跌，以止损价卖出
                        if self.df.loc[i, 'low'] < stop_lose_price:
                            position = 0
                            # 收益使用当前时段HPR
                            self.df.loc[i, 'return'] = stop_lose_price / long_open_price - 1
                        # 持有到当前时段末
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
        self.df['pct_change'].fillna(0, inplace=True)

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
        self.s_return = self.df['return'].mean()
        self.s_std = self.df['return'].std()
        self.s_SR = self.s_return / self.s_std
        self.f_return = self.df['pct_change'].mean()
        self.f_std = self.df['pct_change'].std()
        self.f_SR = self.f_return / self.f_std


class Data4h(Data):
    """
    time interval: 4 hours (240min)
    """

    def __init__(self, csv_file_name, param1, param2):
        super().__init__(csv_file_name, param1, param2)
        self.freq = 240
        self.effect_row = 18

    def add_ranges(self):
        super().add_ranges()
        self.df['high_48h'] = self.df['high'].rolling(12).max()
        self.df['low_48h'] = self.df['low'].rolling(12).min()

        self.df['high_72h'] = self.df['high'].rolling(18).max()
        self.df['low_72h'] = self.df['low'].rolling(18).min()

        self.df['R'] = self.df['high_48h'] - self.df['low_48h']
        self.df['R2'] = self.df['high_72h'] - self.df['low_72h']


class Data1D(Data):
    """
    time interval: 1 day (1440min)
    """

    def __init__(self, csv_file_name, param1, param2):
        super().__init__(csv_file_name, param1, param2)
        self.freq = 1440
        self.effect_row = 18

    def add_ranges(self):
        super().add_ranges()

        self.df['high_5d'] = self.df['high'].rolling(5).max()
        self.df['low_5d'] = self.df['low'].rolling(5).min()

        self.df['R'] = self.df['high'] - self.df['low']
        self.df['R2'] = self.df['high_5d'] - self.df['low_5d']
