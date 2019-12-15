# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20181215"

"""
This is the original program for single factor test for my graduation thesis in Finance Dep. WHU.
You have to change some params by hand. 
Do not use this one; but the codes are correct. 
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pchsale_pchrect=pd.read_excel('64_pchsale_pchrect_final.xlsx')
ret=pd.read_csv('final_return_199701.csv')
size=pd.read_csv('size_199612.csv')
rf=pd.read_csv('RF_199701.csv')#无风险rf
#填充股票代码
df=pd.DataFrame(columns=pchsale_pchrect.columns,index=size.iloc[:,0] )
df=df.fillna(0)
pchsale_pchrect.index=pchsale_pchrect.iloc[:,0]
pchsale_pchrect=pchsale_pchrect+df
pchsale_pchrect=pchsale_pchrect.iloc[:,1:]
pchsale_pchrect.to_csv('64_pchsale_pchrect.csv')
#因子描述性统计
m=pchsale_pchrect.iloc[:,1:].describe()
n=np.array(m)
DES=[]
for i in range(len(n)):
    DES.append(np.average(n[i]))
DES=pd.DataFrame(DES,index=m.index,columns=['describe'] )
print(DES)
#多空组合
RETURN_EW=[]
RETURN_VW=[]
for i in range(len(ret.columns)-1):
    #num = acc.iloc[:, 0]
    Pchsale_pchrect = pchsale_pchrect.iloc[:,i+1]
    RET = ret.iloc[:,i+1]
    Size=size.iloc[:,i+1]
    total = pd.concat([Pchsale_pchrect,RET,Size], axis=1)
    total.columns = ['pchsale_pchrect', 'ret','size']
    total = total.dropna(axis=0)
    total = total.sort_values(by='pchsale_pchrect')
    #equal weighted
    low =total.iloc[0:int(len(total) * 0.1), -2]
    high = total.iloc[-int(len(total) * 0.1):, -2]
    r_final = (sum(high) - sum(low)) / len(high)
    RETURN_EW.append(r_final)
    #value weighted
    low_size=total.iloc[0:int(len(total)*0.1),-1]
    high_size=total.iloc[-int(len(total) * 0.1):,-1]
    #变为可以进行计算的数组
    low=np.array(low)
    low_size =np.array(low_size)
    high =np.array(high)
    high_size =np.array(high_size)
      #市值加权收益率
    LOW = 0
    for j in range(len(low)):
        a = low[j] * low_size[j] / sum(low_size)
        LOW = LOW + a
    HIGH = 0
    for i in range(len(high)):
        b = high[i] * high_size[i] / sum(high_size)
        HIGH = HIGH +b
    r_final = HIGH - LOW
    RETURN_VW.append(r_final)
#因子检验
#年化收益
RET_VW=np.average(RETURN_VW ) *12
RET_EW=np.average(RETURN_EW ) *12
print('annual-return VW',RET_VW,'EW',RET_EW)
#t-test
T_VW=stats.ttest_1samp(RETURN_VW,0)[0]
T_EW=stats.ttest_1samp(RETURN_EW,0)[0]
print('t-statistic VW',T_VW,'EW',T_EW)
#波动率
STD_VW=np.std(RETURN_VW)*np.sqrt(12)
STD_EW=np.std(RETURN_EW)*np.sqrt(12)
print('volitility VW',STD_VW,'EW',STD_EW)
#累计收益&最大回撤
def MaxDrawdown(return_list):
    RET_ACC = []
    sum=1
    for i in range(len(return_list)):
        sum=sum*(return_list[i]+1)
        RET_ACC.append(sum)
    index_j = np.argmax((np.maximum.accumulate(RET_ACC) - RET_ACC)/np.maximum.accumulate(RET_ACC))
    index_i = np.argmax(RET_ACC[:index_j])
    MDD= (RET_ACC[index_i] - RET_ACC[index_j])/RET_ACC[index_i]
    return sum,MDD,RET_ACC
MDD_VW=MaxDrawdown(RETURN_VW)[1]
MDD_EW=MaxDrawdown(RETURN_EW)[1]
ACC_VW=MaxDrawdown(RETURN_VW)[0]
ACC_EW=MaxDrawdown(RETURN_EW)[0]
print('MaxDrawdown VW',MDD_VW,'EW',MDD_EW)
print('accumulated-return VW',ACC_VW,'EW',ACC_EW)
#多空组合净值图
plt. title('pchsale_pchrect')
plt.plot(MaxDrawdown(RETURN_EW)[2],label='Equal Weighted',color='skyblue')
plt.plot(MaxDrawdown(RETURN_VW)[2],label='Value Weighted')
plt
plt.legend(loc='lower right',labels=['Equal Weighted','Value Weighted'])
plt.show().xticks([0,60,120,180,250],['1997','2004','2009','2014','2018'])
#sharp ratio
def sharp(return_list,std):
    returnew = pd.DataFrame(return_list, columns=['R'],index=rf.index)
    m=pd.concat([returnew.R,rf.RF],axis=1)
    m['adj'] = m.apply(lambda x: x['R'] - x['RF'] / 100, axis=1)
    ret_adj=np.array(m.adj)
    sharpratio = np.average(ret_adj)*12 / std
    return sharpratio
SHARP_VW=sharp(RETURN_VW,STD_VW)
SHARP_EW=sharp(RETURN_EW,STD_EW)
print('sharp-ratio VW',SHARP_VW,'EW',SHARP_EW)
#输出月度收益率序列
p=pd.DataFrame(RETURN_VW,columns=['VW'])
q=pd.DataFrame(RETURN_EW,columns=['EW'])
o=pd.concat([p,q],axis=1)
o.to_csv('64_ret_pchsale_pchrect.csv')