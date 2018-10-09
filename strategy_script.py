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


csv_file_name = 'EURUSD240.csv'
CACHE_FNAME = 'data_cache.json'

k1_lst = list(np.arange(0, 1, 0.01).round(2))

try:
    cache_file = open(CACHE_FNAME, 'r')
    cache_contents = cache_file.read()
    CACHE_DICTION = json.loads(cache_contents)
    cache_file.close()
except:
    CACHE_DICTION = {}


def write_new_returns(k1):
    data = Data(csv_file_name=csv_file_name, k1=k1, k2=0.32)
    data.cpt_r_std_SR()
    key = str(k1)
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


def get_returns_using_cache(param_lst):
    s_return_lst = []
    f_return_lst = []
    for k1 in param_lst:
        key = str(k1)
        if key not in CACHE_DICTION:
            print(k1)
            write_new_returns(k1)

        s_return_lst.append(CACHE_DICTION[key]['strategy_return'])
        f_return_lst.append(CACHE_DICTION[key]['forex_return'])
    return s_return_lst, f_return_lst


strategy_return_lst, forex_return_lst = get_returns_using_cache(k1_lst)

plt.plot(strategy_return_lst)
plt.show()
