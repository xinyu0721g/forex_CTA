import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import seaborn as sns

"""
***************************** inputs *****************************************************
"""
CSV_FNAME = 'EURJPY5.csv'
TIME_INT = 5
START = '5/29/2018'
END = '8/31/2018'

R1_LEN = 5
R2_LEN = 10

K1_LST = list(np.arange(0.1, 0.5, 0.1).round(1))
K2_LST = list(np.arange(0.5, 1, 0.1).round(1))
"""
******************************************************************************************
"""
R1_MUL = int(R1_LEN / TIME_INT)
R2_MUL = int(R2_LEN / TIME_INT)
strategy_start_row = int(R2_MUL/R1_MUL)


def create_df_test(csv_file_name=CSV_FNAME, start_date=START, end_date=END):
    df = pd.DataFrame(pd.read_csv(csv_file_name, header=None))
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close']
    start_index = df[(df.date == start_date)].index[0]
    end_index = df[(df.date == end_date)].index[-1]
    df_test = df.loc[start_index:end_index, :]
    return df_test


def create_df_resample(df, r1_multiplier=R1_MUL):
    df_resample = pd.DataFrame(columns=['date', 'time', 'open', 'high', 'low', 'close'])
    for i in np.arange(0, 1000, r1_multiplier):
        row_index_range_start = int(i)
        date, time, open = df.loc[row_index_range_start, ['date', 'time', 'open']]
        row_index_range_end = row_index_range_start + r1_multiplier - 1
        high = df['high'].rolling(r1_multiplier).max()[row_index_range_end]
        low = df['low'].rolling(r1_multiplier).min()[row_index_range_end]
        close, = df.loc[row_index_range_end, ['close']]
        df_resample.loc[row_index_range_start, ] = [date, time, open, high, low, close]
    df_resample.reset_index(drop=True, inplace=True)
    return df_resample


def create_df_for_strategy(csv_file_name=CSV_FNAME, start_date=START, end_date=END, r1_multiplier=R1_MUL):
    df_total = create_df_test(csv_file_name, start_date, end_date)
    df_strategy = create_df_resample(df_total, r1_multiplier)
    return df_strategy


def df_process(df, param1, r1_multiplier=R1_MUL, r2_multiplier=R2_MUL):
    window = int(r2_multiplier/r1_multiplier)
    df['R_high'] = df['high'].rolling(window).max()
    df['R_low'] = df['low'].rolling(window).min()
    df['R'] = df['R_high'] - df['R_low']
    df['long_limit'] = df['open'] - param1 * df['R'].shift(1)


def strategy(df, param2, start_row=strategy_start_row):
    df['pct_change'] = df['close'] / df['open'] - 1
    position = 0
    trigger = 0
    for i in range(start_row, len(df)):

        # 无仓位进行开仓
        if position == 0:
            # if df.R2_low_limit <= df.df.loc[i - 1, 'R2'] <= df.R2_high_limit:

            # 如果最低价低于限价单价格，触及long_limit则买入
            if df.loc[i, 'low'] <= df.loc[i, 'long_limit']:
                trigger = 1
                position = 1
                long_open_price = df.loc[i, 'long_limit']
                stop_win_price = long_open_price + param2 * df.loc[i - 1, 'R']
                stop_lose_price = long_open_price - 0.0050

                # 如果最低价低于止损价格，则说明买入后价格继续下跌，以止损价卖出
                if df.loc[i, 'low'] < stop_lose_price:
                    position = 0
                    # 收益使用当前时段HPR
                    df.loc[i, 'return'] = stop_lose_price / long_open_price - 1
                # 持有到当前时段末
                else:
                    df.loc[i, 'return'] = df.loc[i, 'close'] / long_open_price - 1

        # 有仓位进行平仓
        else:

            # 如果最低价低于止损价格，则平仓
            if df.loc[i, 'low'] < stop_lose_price:
                position = 0
                df.loc[i, 'return'] = min(stop_lose_price, df.loc[i, 'open']) / df.loc[i, 'open'] - 1
            elif df.loc[i, 'high'] > stop_win_price:
                position = 0
                df.loc[i, 'return'] = max(stop_win_price, df.loc[i, 'open']) / df.loc[i, 'open'] - 1
            else:
                df.loc[i, 'return'] = df.loc[i, 'close'] / df.loc[i, 'open'] - 1

        if trigger == 0:
            df['return'] = 0.

    df['return'].fillna(0, inplace=True)
    df['pct_change'].fillna(0, inplace=True)
    df['s_return_cum'] = (df['return']+1).cumprod()
    df['f_return_cum'] = (df['pct_change']+1).cumprod()


def calc_return(df, param1, param2):
    df_process(df, param1)
    strategy(df, param2)
    s_return, = df.loc[len(df)-1, ['s_return_cum']]
    f_return, = df.loc[len(df)-1, ['f_return_cum']]
    return s_return, f_return


CACHE_FNAME = 'cache_{}_{}-{}.json'.format(CSV_FNAME[:-4], R1_LEN, R2_LEN)


try:
    cache_file = open(CACHE_FNAME, 'r')
    cache_contents = cache_file.read()
    CACHE_DICTION = json.loads(cache_contents)
    cache_file.close()
except:
    CACHE_DICTION = {}


def params_unique_combination(param1, param2):
    return '{}_{}'.format(param1, param2)


def write_new_returns(df, param1, param2):
    s_return, f_return = calc_return(df, param1, param2)
    key = params_unique_combination(param1, param2)
    CACHE_DICTION[key] = {}
    CACHE_DICTION[key]['strategy_return'] = s_return
    CACHE_DICTION[key]['forex_return'] = f_return

    dumped_json_cache = json.dumps(CACHE_DICTION, indent=2)
    fw = open(CACHE_FNAME, 'w')
    fw.write(dumped_json_cache)
    fw.close()


def get_return_using_cache(param1, param2):
    key = params_unique_combination(param1, param2)
    if key not in CACHE_DICTION:
        print(param1, param2)
        df = create_df_for_strategy()
        write_new_returns(df, param1, param2)
    strategy_return = CACHE_DICTION[key]['strategy_return']
    return strategy_return


def get_return_df_using_cache(param1_lst, param2_lst):
    """
    先尝试扫描k1
    :param param1_lst: list of param1 to be tested
    :param param2_lst: list of param2 to be tested
    :return: pandas DataFrame, columns are param1s and indexes are param2s
    """
    df_returns = pd.DataFrame()
    for param1 in param1_lst:
        series1_s = pd.Series()
        for param2 in param2_lst:
            s_return = get_return_using_cache(param1, param2)
            series1_s[param2] = s_return
        df_returns[param1] = series1_s
    return df_returns


def plot_strategy(df, param1, param2, r1_multiplier=R1_MUL, r2_multiplier=R2_MUL, start_row=strategy_start_row):
    df_process(df, param1, r1_multiplier, r2_multiplier)
    strategy(df, param2, start_row)
    matplotlib.style.use('ggplot')
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.f_return_cum)
    ax.plot(df.s_return_cum)
    plt.legend()
    plt.show()
    pass


def plot_return_heatmap(param1_lst, param2_lst):
    df = get_return_df_using_cache(param1_lst, param2_lst)
    sns.heatmap(df)
    plt.show()


if __name__ == '__main__':
    # plot_return_heatmap(K1_LST, K2_LST)
    df = create_df_for_strategy()
    plot_strategy(df, 0.3, 0.6)
