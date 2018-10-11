import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


def get_multipliers(r1_length, r2_length, time_int=5):
    """
    get number of intervals/rows for one range (since the dataset is in 5 min, 1 hour would be 12 rows)
    :param r1_length: length of range 1 in minutes
    :param r2_length: length of range 2 in minutes
    :param time_int: length of time interval in minutes
    :return: number of multiples for range 1 and 2
    """
    r1_multi = int(r1_length / time_int)
    r2_multi = int(r2_length / time_int)
    return r1_multi, r2_multi


def data_process(df, r1_multi, r2_multi):
    """
    add columns to DataFrame (R1, R2, R2_past)
    :param df: DataFrame of forex info
    :param r1_multi: int(r1_length/5)
    :param r2_multi: int(r2_length/5)
    :return: None
    """
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close']

    df['high_R1'] = df['high'].rolling(r1_multi).max()
    df['low_R1'] = df['low'].rolling(r1_multi).min()

    df['high_R2'] = df['high'].rolling(r2_multi).max()
    df['low_R2'] = df['low'].rolling(r2_multi).min()

    df['R1'] = df['high_R1'] - df['low_R1']
    df['R2'] = df['high_R2'] - df['low_R2']
    df['R2_past'] = df['R2'].shift(r1_multi)


def gen_x_y(df, r1_multi, r2_multi):
    """
    用历史波动率预测未来波动率
    :param df: DataFrame of forex info
    :param r1_multi: int(r1_length/5)
    :param r2_multi: int(r2_length/5)
    :return: two series of ranges
    """
    x_local = df.R2_past[r1_multi+r2_multi:]
    y_local = df.R1[r1_multi+r2_multi:]
    return x_local, y_local


def calc_corr(x_local, y_local):
    """
    calculate correlation of two time series
    :param x_local: series x
    :param y_local: series y
    :return: correlation of x and y
    """
    return x_local.corr(y_local)


def calc_corr_with_length(df, r1_length, r2_length, time_int=5):
    """
    calculate the needed correlation with DataFrame and range lengths
    :param df: DataFrame of forex info
    :param r1_length: length of range 1 in minutes
    :param r2_length: length of range 2 in minutes
    :param time_int: length of time interval in minutes
    :return: correlation of volatility
    """
    r1_m, r2_m = get_multipliers(r1_length, r2_length, time_int)
    data_process(df, r1_m, r2_m)
    x, y = gen_x_y(df, r1_m, r2_m)
    corr = calc_corr(x, y)
    return corr


def calc_corr_matrix(csv_file_name, r1_lst, r2_lst, time_int=5):
    """
    calculate volatility correlation depends on range1 and range2
    :param csv_file_name: csv file that contains forex info
    :param r1_lst: list of lengths of range 1
    :param r2_lst: list of lengths of range 2
    :param time_int: length of time interval in minutes
    :return: DataFrame of correlation matrix
    """
    dict_local = {}
    for r1 in r1_lst:
        dict_local[r1] = []
        for r2 in r2_lst:
            df = pd.DataFrame(pd.read_csv(csv_file_name, header=None))
            corr_local = calc_corr_with_length(df, r1, r2, time_int)
            dict_local[r1].append(corr_local)
    corr_matrix_local = pd.DataFrame(dict_local)
    corr_matrix_local.set_index([r2_lst], inplace=True)
    return corr_matrix_local


def fit_data_OLS(x_local, y_local):
    """
    fit linear regression using OLS
    :param x_local: series x
    :param y_local: series y
    :return: None, but show OLS summary and graph
    """
    x_local = sm.add_constant(x_local)
    est = sm.OLS(y_local, x_local)
    est = est.fit()
    print(est.summary())

    x_prime = np.linspace(x_local.R2_past.min(), x_local.R2_past.max(), 100)
    x_prime = sm.add_constant(x_prime)
    y_hat = est.predict(x_prime)
    plt.scatter(x_local.R2_past, y_local, alpha=0.3)
    plt.xlabel('R1')
    plt.ylabel('R2')
    plt.plot(x_prime[:, 1], y_hat, 'r', alpha=0.9)
    plt.show()


def fit_data_with_length(df, r1_length, r2_length, time_int=5):
    """
    a comprehensive function that uses initial inputs to get OLS fit
    :param df: DataFrame of forex info
    :param r1_length: length of range 1 in minutes
    :param r2_length: length of range 2 in minutes
    :param time_int: length of time interval in minutes
    :return: None, but show OLS summary and graph
    """
    r1_m, r2_m = get_multipliers(r1_length, r2_length, time_int)
    data_process(df, r1_m, r2_m)
    x, y = gen_x_y(df, r1_m, r2_m)
    corr = calc_corr(x, y)
    print(corr)
    fit_data_OLS(x, y)


def plot_heatmap(data):
    """
    plot heatmap
    :param data: DataFrame used as data input in heatmap
    :return: None. But show heatmap graph
    """
    sns.heatmap(data)
    plt.show()


if __name__ == "__main__":

    """
    inputs: CSV_FNAME, R1_lst, R2_lst
    """
    CSV_FNAME = 'EURJPY5.csv'
    time_interval = 5
    R1_lst = [5, 10, 30, 60, 120, 240, 480, 1440]
    R2_lst = [60, 120, 240, 480, 1440, 2880]

    """
    calculate correlation with specific lengths
    """
    R1_length = 10
    R2_length = 60
    DF = pd.DataFrame(pd.read_csv(CSV_FNAME, header=None))
    # corr = calc_corr_with_length(DF, R1_length, R2_length)
    # print(corr)
    fit_data_with_length(DF, R1_length, R2_length)

    """
    calculate correlation matrix
    """
    # corr_matrix = calc_corr_matrix(CSV_FNAME, R1_lst, R2_lst, time_interval)
    # print(corr_matrix)
    # plot_heatmap(corr_matrix)
