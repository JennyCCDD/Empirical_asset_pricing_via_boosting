# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20190329"
"""
This is the original main program for my graduation thesis in Finance Dep. WHU.
There are some problems for this program:
1) the porfilios are calculated based on the risk premium;
2) it's difficult to seperate the loop into several function and put all of them into a class;
3) FM(OLS) is tend to come across data leakage and produce extreme value for MSE and R^2；
4) ...
"""

import numpy as np
import pandas as pd
import datetime
import os
from sklearn import linear_model
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import gc
import warnings
warnings.filterwarnings("ignore")
starttime = datetime.datetime.now()

class Para:
    method = 'FM'    #method_list = ['FM','EXT','GBRT','XGB','LBGM']
    portfolioform ='all'
    time_window = 12 #months
    rollingtime = 1  #months
    path_data = '.\\normalized\\'
    path = '.\\input\\'
    path_results = '.\\results\\'
    long_short_ratio = 0.1
para = Para()

rr_input = pd.read_csv(para.path + 'risk_premium_199701.csv', index_col=0, parse_dates=True)
size_input = pd.read_csv(para.path + 'size_199612.csv', index_col=0, parse_dates=True)
ret_input = pd.read_csv(para.path + 'final_return_199701.csv', index_col=0, parse_dates=True)
rf_input = pd.read_csv(para.path + 'rf_199701.csv', index_col=0, parse_dates=True)
rr = rr_input.iloc[:, para.time_window:].copy()
# get number of month, stockis, datelist
number_of_month = rr_input.columns.size
stockid = rr_input.index.tolist()
datelist = rr.columns.tolist()
datelist.pop(0)
RR = rr_input.iloc[:, para.time_window:].copy()
Size = size_input.iloc[:, para.time_window:].copy()
ret = ret_input.iloc[:, para.time_window:].copy()
rf = rf_input.iloc[:, (para.time_window + 1):].copy()

Mean_squared_error = []
R_2 = []
RR_EW = []
RR_VW = []
RR_EW_Long = []
RR_EW_Short = []
RR_VW_Long = []
RR_VW_Short = []
COEF = []
for k in range(int(para.time_window + 1), number_of_month, para.rollingtime):
    y_score_out_of_sample_many=[]
    for i in range(k - para.time_window, k, para.rollingtime):
        file_name = para.path_data + str(i) + '.csv'
        data_curr_month = pd.read_csv(file_name, header=0, low_memory=False)
        data_curr_month = data_curr_month.dropna(axis=0, how='all')
        para.n_stock = data_curr_month.shape[0]
        data_in_sample = data_curr_month.fillna(0)
        X_in_sample = data_in_sample.loc[:, 'size':'sin']
        y_in_sample = data_in_sample.loc[:, 'ret']
        features = pd.DataFrame(X_in_sample.columns)
        file_name2 = para.path_data + str(k) + '.csv'
        data_next_month = pd.read_csv(file_name2, header=0, low_memory=False)
        data_next_month = data_next_month.dropna(axis=0, how='all')
        data_out_of_sample = data_next_month.fillna(0)
        X_out_of_sample = data_out_of_sample.loc[:, 'size':'sin']
        y_out_of_sample = data_out_of_sample.loc[:, 'ret']
        if para.method == 'FM':
            model = linear_model.LinearRegression()  # fit_intercept=True, normalize=False, copy_X=True)
        elif para.method == 'EXT':
            model = ExtraTreesRegressor()
        elif para.method == 'GBDT':
            model = GradientBoostingRegressor()
        elif para.method == 'XGB':
            model = xgb.XGBRegressor(silent=True)
        elif para.method == 'LBGM':
            model = lgb.LGBMRegressor(silent=False)
        else:
            pass
        model.fit(X_in_sample, y_in_sample)
        # model predict
        y_score_out_of_sample = model.predict(X_out_of_sample)
        y_score_out_of_sample_many.append(y_score_out_of_sample)
    y_score_out_of_sample_many_df = pd.DataFrame(y_score_out_of_sample_many)
    y_score_out_of_sample_average = []
    for x in range(y_score_out_of_sample_many_df.columns.size):
        y_score_out_of_sample_ave = np.average(y_score_out_of_sample_many_df[x])
        y_score_out_of_sample_average.append(y_score_out_of_sample_ave)
    y_score_out_of_sample_average = pd.DataFrame(y_score_out_of_sample_average)
    # output the coefficent
    if not os.path.exists(para.path_results):
        os.makedirs(para.path_results)
    if para.method == 'FM':
        coef = pd.DataFrame(model.coef_)
    else:
        coef = pd.DataFrame(model.feature_importances_)
    COEF.append(coef)
    model_coef = pd.merge(features, coef, left_index=True, right_index=True)
    filename_coef = para.path_results + 'coef_' + '%s' % Para.method + '_' + '%d' % (k) + '.csv'
    model_coef.to_csv(filename_coef)
    # form the portfolio
    RR = rr.iloc[:, k - para.time_window].copy()
    SIZE = Size.iloc[:, k - para.time_window].copy()
    total = pd.concat([y_score_out_of_sample_average, RR, SIZE], axis=1)
    total = total.astype('float')
    total.columns = ['prediction', 'risk_premium', 'size']
    if not os.path.exists(para.path_results):
        os.makedirs(para.path_results)
    filename_prediction = para.path_results + 'model_prediction' + '%s' % Para.method + '_' + '%d' % (
            k + 1) + '.csv'
    total.to_csv(filename_prediction)
    total = total.dropna(axis=0)

    if para.portfolioform == 'top':
        total_portfolioform = total.sort_values(by='size', ascending=False)
        total_portfolioform = total.iloc[0:int(len(total) * 0.8), :]
    if para.portfolioform == 'bottom':
        total_portfolioform = total.sort_values(by='size', ascending=False)
        total_portfolioform = total.iloc[-int(len(total) * 0.8):0, :]
    else:
        total_portfolioform = total
    total_sort = total_portfolioform.sort_values(by='prediction')
    # equal weighted
    low = total_sort.iloc[0:int(len(total) * para.long_short_ratio), -2]#'risk_premium'
    high = total_sort.iloc[-int(len(total) * para.long_short_ratio):, -2]#'risk_premium'
    r_final = (sum(high) - sum(low)) / len(high)
    r_long = sum(high) / len(high)
    r_short = sum(low) / len(high)
    RR_EW.append(r_final)
    RR_EW_Long.append(r_long)
    RR_EW_Short.append(r_short)
    # value weighted
    low_size = total_sort.iloc[0:int(len(total) * para.long_short_ratio), -1]#'size'
    high_size = total_sort.iloc[-int(len(total) * para.long_short_ratio):, -1]#'size'
    low = np.array(low)
    low_size = np.array(low_size)
    high = np.array(high)
    high_size = np.array(high_size)
    LOW = 0
    for j in range(len(low)):
        a = low[j] * low_size[j] / sum(low_size)
        LOW = LOW + a
    HIGH = 0
    for jj in range(len(high)):
        b = high[jj] * high_size[jj] / sum(high_size)
        HIGH = HIGH + b
    r_final = HIGH - LOW
    r_long = HIGH
    r_short = LOW
    RR_VW.append(r_final)
    RR_VW_Long.append(r_long)
    RR_VW_Short.append(r_short)
    # evaluate the model
    total_sort = total.sort_values(by='prediction')
    MSE = mean_squared_error(total_sort['risk_premium'], total_sort['prediction'])
    r_2 = r2_score(total_sort['risk_premium'], total_sort['prediction'])
    print('test set, month%d,MSE =%.5f' % (k, MSE))
    print('test set, month%d,R^2=%.5f' % (k, r_2))
    Mean_squared_error.append(mean_squared_error)
    R_2.append(r_2)
    gc.collect()
MSE = np.average(Mean_squared_error)
R2 = np.average(R_2)
print('Out of sample MSE', MSE)
print('Out of sample R2', R2)
if not os.path.exists(para.path_results):
    os.makedirs(para.path_results)
filename = para.path_results + 'statistics_' + '%s' % Para.method + '.csv'
p = pd.DataFrame(Mean_squared_error, columns=['Mean_squared_error'])
q = pd.DataFrame(R_2, columns=['R_2'])
o = pd.concat([p, q], axis=1)
o.to_csv(filename)
# 输出月度收益率序列
if not os.path.exists(para.path_results):
    os.makedirs(para.path_results)
filename = para.path_results + 'monthly RISK PREMIUM FINAL_' + '%s' % Para.method + '.csv'
r_EW = pd.DataFrame(RR_EW, columns=['EW_long_short'])
r_EW_Long = pd.DataFrame(RR_EW_Long, columns=['EW_long'])
r_EW_Short = pd.DataFrame(RR_EW_Short, columns=['EW_short'])
r_VW = pd.DataFrame(RR_VW, columns=['VW_long_short'])
r_VW_Long = pd.DataFrame(RR_VW_Long, columns=['VW_long'])
r_VW_Short = pd.DataFrame(RR_VW_Short, columns=['VW_short'])
o1 = pd.concat([r_EW, r_EW_Long, r_EW_Short, r_VW, r_VW_Long, r_VW_Short], axis=1)
o1.to_csv(filename)
# 由机器学习算法构造的多空组合
# 年化收益
RET_VW = np.average(RR_VW) * 12
RET_EW = np.average(RR_EW) * 12
print('annual-return VW', RET_VW, 'EW', RET_EW)
# t-test
T_VW = stats.ttest_1samp(RR_VW, 0)[0]
T_EW = stats.ttest_1samp(RR_EW, 0)[0]
print('t-statistic VW', T_VW, 'EW', T_EW)
# 波动率
STD_VW = np.std(RR_VW) * np.sqrt(12)
STD_EW = np.std(RR_EW) * np.sqrt(12)
print('volitility VW', STD_VW, 'EW', STD_EW)

# 累计收益&最大回撤
def MaxDrawdown(return_list):
    RET_ACC = []
    sum = 1
    for i in range(len(return_list)):
        sum = sum * (return_list[i] + 1)
        RET_ACC.append(sum)
    index_j = np.argmax((np.maximum.accumulate(RET_ACC) - RET_ACC) / np.maximum.accumulate(RET_ACC))
    index_i = np.argmax(RET_ACC[:index_j])
    MDD = (RET_ACC[index_i] - RET_ACC[index_j]) / RET_ACC[index_i]
    return sum, MDD, RET_ACC

MDD_VW = MaxDrawdown(RR_VW)[1]
MDD_EW = MaxDrawdown(RR_EW)[1]
ACC_VW = MaxDrawdown(RR_VW)[0]
ACC_EW = MaxDrawdown(RR_EW)[0]
print('MaxDrawdown VW', MDD_VW, 'EW', MDD_EW)
print('Accumulated risk premium VW', ACC_VW, 'EW', ACC_EW)

# sharp ratio
def sharp(return_list, std):
    returnew = pd.DataFrame(return_list, columns=['R'])
    m = pd.concat([returnew.R], axis=1)
    ret_adj = np.array(m)
    sharpratio = np.average(ret_adj) * 12 / std
    return sharpratio

SHARP_VW = sharp(RR_VW, STD_VW)
SHARP_EW = sharp(RR_EW, STD_EW)
print('sharp-ratio VW', SHARP_VW, 'EW', SHARP_EW)

# 计算程序运行时间
endtime = datetime.datetime.now()
def timeStr(s):
    if s < 10:
        return '0' + str(s)
    else:
        return str(s)
print("程序开始运行时间：" + timeStr(starttime.hour) + ":" + timeStr(starttime.minute) + ":" + timeStr(starttime.second))
print("程序结束运行时间：" + timeStr(endtime.hour) + ":" + timeStr(endtime.minute) + ":" + timeStr(endtime.second))
runTime = (endtime - starttime).seconds
runTimehour = runTime // 3600  # 除法并向下取整，整除
runTimeminute = (runTime - runTimehour * 3600) // 60
runTimesecond = runTime - runTimehour * 3600 - runTimeminute * 60
print("程序运行耗时：" + str(runTimehour) + "时" + str(runTimeminute) + "分" + str(runTimesecond) + "秒")
