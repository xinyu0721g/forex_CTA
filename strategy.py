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

参数扫描：
- 限价单参数k1
- 止盈参数k2
"""

from classes import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json

"""
*************************** data set and params *****************************************
"""
CSV_FNAME = 'EURJPY5.csv'
TI = 1440  # to determine which kind of Data objects to create
k1_lst = list(np.arange(0, 1, 0.01).round(2))
k2_lst = list(np.arange(0, 1, 0.01).round(2))

"""
*************************** get returns using cache *************************************
"""
CACHE_FNAME = 'cache_{}.json'.format(CSV_FNAME[:-4])

try:
    cache_file = open(CACHE_FNAME, 'r')
    cache_contents = cache_file.read()
    CACHE_DICTION = json.loads(cache_contents)
    cache_file.close()
except:
    CACHE_DICTION = {}


def params_unique_combination(param1, param2):
    return '{}_{}'.format(param1, param2)


def create_data_obj(param1, param2, csv_file_name=CSV_FNAME, time_interval=TI):
    if time_interval == 240:
        df_local = Data4h(csv_file_name=csv_file_name, param1=param1, param2=param2)
    elif time_interval == 1440:
        df_local = Data1D(csv_file_name=csv_file_name, param1=param1, param2=param2)
    else:
        df_local = None
    return df_local


def write_new_returns(param1, param2, csv_file_name=CSV_FNAME, time_interval=TI):
    data = create_data_obj(param1, param2, csv_file_name, time_interval)
    data.cpt_r_std_SR()
    key = params_unique_combination(param1, param2)
    CACHE_DICTION[key] = {}
    CACHE_DICTION[key]['strategy_return'] = data.s_return
    CACHE_DICTION[key]['strategy_std'] = data.s_std
    CACHE_DICTION[key]['strategy_SR'] = data.s_SR
    CACHE_DICTION[key]['forex_return'] = data.f_return
    CACHE_DICTION[key]['forex_std'] = data.f_std
    CACHE_DICTION[key]['forex_SR'] = data.f_SR

    dumped_json_cache = json.dumps(CACHE_DICTION, indent=2)
    fw = open(CACHE_FNAME, 'w')
    fw.write(dumped_json_cache)
    fw.close()


def get_return_using_cache(param1, param2):
    key = params_unique_combination(param1, param2)
    if key not in CACHE_DICTION:
        print(param1, param2)
        write_new_returns(param1, param2)
    strategy_return = CACHE_DICTION[key]['strategy_return']
    return strategy_return


def get_returns_using_cache(param1_lst, param2_lst):
    """
    先尝试扫描k1
    :param param1_lst: list of param1 to be tested
    :param param2_lst: list of param2 to be tested
    :return: pandas DataFrame, columns are param1s and indexes are param2s
    """
    df = pd.DataFrame()
    for param1 in param1_lst:
        series1_s = pd.Series()
        for param2 in param2_lst:
            s_return = get_return_using_cache(param1, param2)
            series1_s[param2] = s_return
        df[param1] = series1_s
    return df


def plot_return_heatmap(param1_lst, param2_lst):
    df = get_returns_using_cache(param1_lst, param2_lst)
    pass


if __name__ == '__main__':
    df = get_returns_using_cache(k1_lst, k2_lst)
    print(df)
