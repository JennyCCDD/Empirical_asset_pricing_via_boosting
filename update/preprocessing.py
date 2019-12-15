# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191214"

"""
this programe is really slow, though the codes are consise
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
import os

class Para:
    path_data = '.\\factors\\'
    path = '.\\input\\'
    path_results = '.\\output\\'
para = Para()

class _preprocessing_:
    def __init__(self):
        return
    def file_name(self):
        path = r"E:\毕业论文\test\re\input\factors"
        F = []
        for root, dirs, files in os.walk(path):
            for file in files:
                #print file.decode('gbk')    #recode if there's Chinese
                if os.path.splitext(file)[1] == '.csv':
                    t = os.path.splitext(file)[0]
                    F.append(t)
        return F
    def data_get(self):
        ret = pd.read_csv(para.path + 'risk_premium_199701.csv', low_memory=False)
        stockid = pd.read_csv(para.path + 'stockid.csv', low_memory=False)
        industryid = pd.read_csv(para.path + 'industryid.csv', low_memory=False)
        ret = ret.iloc[:, 1:]
        stockid = stockid['stockid']
        industryid = industryid['industryid']
        return ret,stockid,industryid

if __name__ == '__main__':
    factor_list = _preprocessing_().file_name()
    ret,stockid,industryid = _preprocessing_().data_get()

    factors = factor_list.copy()
    factors.insert(0, "stockid")
    factors.insert(1, "industryid")
    factors.insert(2, "ret")

    outputname3 = para.path_results + '截面因子数据.xlsx'
    with pd.ExcelWriter(outputname3) as writer3:

        for j in range(len(ret.columns) - 1):
            RET = ret.iloc[:, j]
            flist = []
            for i in factor_list:
                filename = para.path_data + i + '.csv'
                factor_i = pd.read_csv(filename)#, low_memory=False)
                factor_i_pre = factor_i.iloc[:, 1:]
                factor_i_arr = np.array(factor_i_pre)
                quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                                         random_state=0)
                factor_i_ = quantile_transformer.fit_transform(factor_i_arr)
                factor_i_df = pd.DataFrame(factor_i_)
                factor_i_df = factor_i_df.iloc[:, j]
                flist.append(factor_i_df)
            flist_df = pd.DataFrame(flist,columns = stockid)
            flist_df.fillna(flist_df.mean(axis = 0),axis = 0,inplace = True)
            flist_df_T = flist_df.T
            flist_df_T.columns = factor_list
            print(flist_df_T)
            other = pd.concat([stockid, industryid, RET], axis=1)
            other.set_index (other['stockid'],inplace = True)
            print(other)
            total = pd.concat([other,flist_df_T],axis = 1)
            total.columns = factors
            total.set_index(total["stockid"], inplace=True)
            print(total)

            sheetname3 = '%s' % i
            print(sheetname3)
            total.to_excel(writer3, sheet_name=i)

