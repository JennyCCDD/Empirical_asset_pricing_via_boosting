# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191128"

import numpy as np

# define the function to form thw portfolios of eqaul weighted
# input: dataframe: total.columns = ['prediction', 'real_return', 'size', 'risk_premium']
# output: dataframe: total_sort_.columns = ['prediction', 'real_return', 'size', 'risk_premium',
#                                            'flag','weight','portfolio_return']
#         float: long_ew(return of long side)
#                short_ew(return of short side)
#                long_short_ew(return of long minus short)
def portfolio_EW(total,long_short_ratio):
    total.dropna(axis=0, inplace=True)
    """
    def cap(x, quantile=[0.01, 0.99]):# 生成分位数
        Q01, Q99 = x.quantile(quantile).values.tolist()# 替换异常值为指定的分位数
        if Q01 > x.min():
            x = x.copy()
            x.loc[x < Q01] = Q01
        if Q99 < x.max():
            x = x.copy()
            x.loc[x > Q99] = Q99
        return (x)
    #total = cap(total)
    #for index, row in total.iterrows():
        #if (row >= (row-row.mean())/row.std()):
            #total.loc[index] = total.mean()#.astype(int)
    #total = total[['prediction', 'real_return', 'size', 'risk_premium']]
    #total.apply(lambda x: x.replace(a,x.mean(axis = 1)) for a-x.mean()>3*x.std() , axis=1)
    """
    #total.drop('size', 1, inplace=True)
    total_sort_ = total.sort_values(by='prediction',ascending=True) # 默认升序
    total_sort_.columns = total.columns

    pre_arr = np.array(total_sort_['prediction'])
    ratio_long = np.percentile(pre_arr, (1 - long_short_ratio))
    ratio_short = np.percentile(pre_arr, long_short_ratio)

    long_num = len(total_sort_[total_sort_['prediction'] >ratio_long])
    short_num = len(total_sort_[total_sort_['prediction'] <ratio_short])
    if short_num==0:
        short_num =1
    else:
        pass
    total_sort_.loc[total_sort_['prediction'] > ratio_long, 'flag'] = 1
    total_sort_.loc[total_sort_['prediction'] < ratio_short, 'flag'] = -1

    total_sort_.loc[total_sort_['flag'] == 1, 'weight'] = 1 / long_num
    total_sort_.loc[total_sort_['flag'] == -1, 'weight'] = 1/ short_num

    total_sort_.loc[total_sort_['flag'] == 1, 'portfolio_return'] = \
        total_sort_.apply(lambda x: x['weight'] * x['real_return'], axis=1)
    total_sort_.loc[total_sort_['flag'] == -1, 'portfolio_return'] = \
        total_sort_.apply(lambda x: x['weight'] * x['real_return'], axis=1)

    long_ew = total_sort_.loc[total_sort_['flag'] == 1, 'portfolio_return'].sum()
    short_ew = total_sort_.loc[total_sort_['flag'] == -1, 'portfolio_return'].sum()
    long_short_ew = long_ew - short_ew
    return total_sort_, long_ew, short_ew, long_short_ew

# define the function to form thw portfolios of value weighted
# input and output are similar to function portfolio_EW
def portfolio_VW(total,long_short_ratio):
    total.dropna(axis=0,inplace=True)
    total_sort_ = total.sort_values(by='prediction')
    total_sort_.columns = total.columns
    pre_arr = np.array(total_sort_['prediction'])
    ratio_long = np.percentile(pre_arr, (1 - long_short_ratio))
    ratio_short = np.percentile(pre_arr, long_short_ratio)

    total_sort_.loc[total_sort_['prediction']>ratio_long, 'flag'] = 1
    total_sort_.loc[total_sort_['prediction']<ratio_short, 'flag'] = -1

    total_size_long = total_sort_.loc[total_sort_['flag']==1,'size'].sum()
    total_size_short = total_sort_.loc[total_sort_['flag']==1,'size'].sum()

    total_sort_.loc[total_sort_['flag'] == 1,'weight'] = total_sort_['size']/total_size_long
    total_sort_.loc[total_sort_['flag'] == -1,'weight'] = total_sort_['size'] / total_size_short

    total_sort_.loc[total_sort_['flag'] == 1,'portfolio_return'] = \
        total_sort_.apply(lambda x: x['weight'] * x['real_return'],axis=1)
    total_sort_.loc[total_sort_['flag'] == -1,'portfolio_return'] = \
        total_sort_.apply(lambda x: x['weight'] * x['real_return'],axis=1)

    long_vw = total_sort_.loc[total_sort_['flag'] == 1,'portfolio_return'].sum()
    short_vw = total_sort_.loc[total_sort_['flag'] == -1,'portfolio_return'].sum()
    long_short_vw = long_vw-short_vw
    return total_sort_,long_vw, short_vw, long_short_vw

# define the function to form thw portfolios of weight set already
# input: dataframe: total.columns = ['prediction', 'real_return', 'size', 'risk_premium','weight']
# output: dataframe: total_sort_.columns = ['prediction', 'real_return', 'size', 'risk_premium','weight',
#                                            'flag','portfolio_return']
#         float: similar to function portfolio_EW
def portfolio_weightset(total,long_short_ratio):
    total.dropna(axis=0,inplace=True)
    total_sort_ = total.sort_values(by='prediction')
    total_sort_.columns = total.columns
    pre_arr = np.array(total_sort_['prediction'])
    ratio_long = np.percentile(pre_arr, (1 - long_short_ratio))
    ratio_short = np.percentile(pre_arr, long_short_ratio)

    total_sort_.loc[total_sort_['prediction']>ratio_long, 'flag'] = 1
    total_sort_.loc[total_sort_['prediction']<ratio_short, 'flag'] = -1

    total_sort_.loc[total_sort_['flag'] == 1,'portfolio_return'] = \
        total_sort_.apply(lambda x: x['weight'] * x['real_return'],axis=1)
    total_sort_.loc[total_sort_['flag'] == -1,'portfolio_return'] = \
        total_sort_.apply(lambda x: -x['weight'] * x['real_return'],axis=1)

    long_ws = total_sort_.loc[total_sort_['flag'] == 1,'portfolio_return'].sum()
    short_ws = total_sort_.loc[total_sort_['flag'] == -1,'portfolio_return'].sum()
    long_short_ws = long_ws+short_ws
    return total_sort_,long_ws, short_ws, long_short_ws
