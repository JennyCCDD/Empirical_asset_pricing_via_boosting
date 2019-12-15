# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__ = "chenmx19@mails.tsinghua.edu.cn"
__date__ = "20191128"

#--* import pakages *--#
import pandas as pd
import numpy as np
import os
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simkai.ttf', size=15)
import pandas_profiling

#--* import pakages I written*--#
from portfolio_construction import portfolio_EW, portfolio_VW
from performance_measure import performance, performance_anl

#--* define para class*--#
class Para:
    path_data = '.\\input\\factors\\'
    path = '.\\input\\'
    path_results = '.\\output\\'
    long_short_ratio = 0.1
para = Para()

class _single_factor_:
    def __init__(self):
        return

    def file_name(self):
        path = r"E:\毕业论文\test\re\input\factors"
        F = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == '.csv':
                    t = os.path.splitext(file)[0]
                    F.append(t)
        return F

    def descriptive(self,df):
        m = factor_i.iloc[:, 1:].describe()
        n = np.array(m)
        DES = []
        for i in range(len(n)):
            DES.append(np.average(n[i]))
        DES = pd.DataFrame(DES, index=m.index, columns=['describe'])
        print(DES)
        return DES

if __name__ == '__main__':
    factor_list = _single_factor_().file_name()

    rr = pd.read_csv(para.path + 'risk_premium_199701.csv', index_col=0, parse_dates=True)
    size = pd.read_csv(para.path + 'size_199612.csv', index_col=0, parse_dates=True)
    ret = pd.read_csv(para.path + 'final_return_199701.csv', index_col=0, parse_dates=True)
    rf = pd.read_csv(para.path + 'rf_199701.csv', index_col=0, parse_dates=True)
    datelist = ret.columns.tolist()
    datelist.pop(0)

    outputname1 = para.path_results + 'factor_descriptive_analysis.xlsx'
    with pd.ExcelWriter(outputname1) as writer1:
        descriptive_list = []
        for a in factor_list:
            filename = para.path_data + a + '.csv'
            factor_i = pd.read_csv(filename, index_col=0, parse_dates=True)
            descriptive_i = _single_factor_().descriptive(factor_i)
            descriptive_list.append(descriptive_i)
            sheetname1 = '%s' % a
            descriptive_i.to_excel(writer1, sheet_name=sheetname1)


    for a in factor_list:
        filename = para.path_data + a + '.csv'
        factor_i = pd.read_csv(filename, index_col=0, parse_dates=True)
        RR_EW = []
        RR_VW = []
        RR_EW_Long = []
        RR_EW_Short = []
        RR_VW_Long = []
        RR_VW_Short = []
        for i in range(len(factor_i.columns) - 1):
            factor_ii = factor_i.iloc[:, i]
            RET = ret.iloc[:, i]
            Size = size.iloc[:, i]
            RR = rr.iloc[:, i]
            total = pd.concat([factor_ii, RET, Size, RR], axis=1)
            total.columns = ['prediction', 'real_return', 'size', 'risk_premium']
            total = total.astype('float')
            total_sort_ew, long_ew, short_ew, long_short_ew = portfolio_EW(total, para.long_short_ratio)
            total_sort_vw, long_vw, short_vw, long_short_vw = portfolio_VW(total, para.long_short_ratio)
            RR_EW_Long.append(long_ew)
            RR_EW_Short.append(short_ew)
            RR_EW.append(long_short_ew)
            RR_VW_Long.append(long_vw)
            RR_VW_Short.append(short_vw)
            RR_VW.append(long_short_vw)

        strategy_RR_EW_Long = pd.DataFrame(RR_EW_Long, columns=['monthly'], index=datelist)
        strategy_RR_EW_Long.index = pd.DatetimeIndex(strategy_RR_EW_Long.index)
        strategy_RR_EW_Long['nav'] = (strategy_RR_EW_Long['monthly'] + 1).cumprod()
        strategy_RR_EW_Long.to_csv(para.path_results + 'strategy_RR_EW_Long_%s' % a + '.csv')
        print('__________________strategy_RR_EW_Long__________________')
        performance(strategy_RR_EW_Long)
        performance_anl(strategy_RR_EW_Long)

        strategy_RR_EW_Short = pd.DataFrame(RR_EW_Short, columns=['monthly'], index=datelist)
        strategy_RR_EW_Short.index = pd.DatetimeIndex(strategy_RR_EW_Short.index)
        strategy_RR_EW_Short['nav'] = (strategy_RR_EW_Short['monthly'] + 1).cumprod()
        strategy_RR_EW_Short.to_csv(para.path_results + 'strategy_RR_EW_Short_%s' % a + '.csv')
        print('__________________strategy_RR_EW_Short__________________')
        performance(strategy_RR_EW_Short)
        performance_anl(strategy_RR_EW_Short)

        strategy_RR_EW = pd.DataFrame(RR_EW, columns=['monthly'], index=datelist)
        strategy_RR_EW.index = pd.DatetimeIndex(strategy_RR_EW.index)
        strategy_RR_EW['nav'] = (strategy_RR_EW['monthly'] + 1).cumprod()
        strategy_RR_EW.to_csv(para.path_results + 'strategy_RR_EW_%s' % a + '.csv')
        print('__________________strategy_RR_EW__________________')
        performance(strategy_RR_EW)
        performance_anl(strategy_RR_EW)

        strategy_RR_VW_Long = pd.DataFrame(RR_VW_Long, columns=['monthly'], index=datelist)
        strategy_RR_VW_Long.index = pd.DatetimeIndex(strategy_RR_VW_Long.index)
        strategy_RR_VW_Long['nav'] = (strategy_RR_VW_Long['monthly'] + 1).cumprod()
        strategy_RR_VW_Long.to_csv(para.path_results + 'strategy_RR_VW_Long_%s' % a + '.csv')
        print('__________________strategy_RR_VW_Long__________________')
        performance(strategy_RR_VW_Long)
        performance_anl(strategy_RR_VW_Long)

        strategy_RR_VW_Short = pd.DataFrame(RR_VW_Short, columns=['monthly'], index=datelist)
        strategy_RR_VW_Short.index = pd.DatetimeIndex(strategy_RR_VW_Short.index)
        strategy_RR_VW_Short['nav'] = (strategy_RR_VW_Short['monthly'] + 1).cumprod()
        strategy_RR_VW_Short.to_csv(para.path_results + 'strategy_RR_VW_Short_%s' % a + '.csv')
        print('__________________strategy_RR_VW_Short__________________')
        performance(strategy_RR_VW_Short)
        performance_anl(strategy_RR_VW_Short)

        strategy_RR_VW = pd.DataFrame(RR_VW, columns=['monthly'], index=datelist)
        strategy_RR_VW.index = pd.DatetimeIndex(strategy_RR_VW.index)
        strategy_RR_VW['nav'] = (strategy_RR_VW['monthly'] + 1).cumprod()
        strategy_RR_VW.to_csv(para.path_results + 'strategy_RR_VW_%s' % a + '.csv')
        print('__________________strategy_RR_VW__________________')
        performance(strategy_RR_VW)
        performance_anl(strategy_RR_VW)