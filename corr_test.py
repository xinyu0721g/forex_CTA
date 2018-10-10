import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


def get_multipliers(r1_length, r2_length):
    r1_multi = int(r1_length/5)
    r2_multi = int(r2_length/5)
    return r1_multi, r2_multi


def data_process(df, r1_multi, r2_multi):
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
    """
    x_local = df.R2_past[r1_multi+r2_multi:]
    y_local = df.R1[r1_multi+r2_multi:]
    return x_local, y_local


def calc_corr(x_local, y_local):
    return x_local.corr(y_local)


def calc_corr_with_length(df, r1_length, r2_length):
    r1_m, r2_m = get_multipliers(r1_length, r2_length)
    data_process(df, r1_m, r2_m)
    x, y = gen_x_y(df, r1_m, r2_m)
    corr = calc_corr(x, y)
    return corr


def calc_corr_matrix(csv_file_name, r1_lst, r2_lst):
    dict_local = {}
    for r1 in r1_lst:
        dict_local[r1] = []
        for r2 in r2_lst:
            df = pd.DataFrame(pd.read_csv(csv_file_name, header=None))
            corr_local = calc_corr_with_length(df, r1, r2)
            dict_local[r1].append(corr_local)
    corr_matrix_local = pd.DataFrame(dict_local)
    corr_matrix_local.set_index([r2_lst], inplace=True)
    return corr_matrix_local


def fit_data_OLS(x_local, y_local):
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


def fit_data_with_length(df, r1_length, r2_length):
    r1_m, r2_m = get_multipliers(r1_length, r2_length)
    data_process(df, r1_m, r2_m)
    x, y = gen_x_y(df, r1_m, r2_m)
    corr = calc_corr(x, y)
    print(corr)
    fit_data_OLS(x, y)


def plot_heatmap(data):
    sns.heatmap(data)
    plt.show()


if __name__ == "__main__":
    CSV_FNAME = 'EURJPY5.csv'

    R1_lst = [5, 10, 30, 60, 120, 240, 480, 1440]
    R2_lst = [60, 120, 240, 480, 1440, 2880]

    # calculate correlation with specific lengths
    # R1_length = 60
    # R2_length = 60
    # DF = pd.DataFrame(pd.read_csv(CSV_FNAME, header=None))
    # r1_m, r2_m = get_multipliers(R1_length, R2_length)
    # data_process(DF, r1_m, r2_m)
    # print(DF.head(50))

    # corr = calc_corr_with_length(DF, R1_length, R2_length)
    # print(corr)

    corr_matrix = calc_corr_matrix(CSV_FNAME, R1_lst, R2_lst)
    print(corr_matrix)

    plot_heatmap(corr_matrix)
