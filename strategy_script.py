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


csv_file_name = 'EURUSD240.csv'

k1_lst = list(np.arange(0, 1, 0.01).round(2))

return_lst = []
for k1 in k1_lst[:10]:
    print(k1)
    data = Data(csv_file_name=csv_file_name, k1=k1, k2=0.32)
    data.cpt_r_std_SR()
    return_lst.append(data.return_)

print(return_lst)
