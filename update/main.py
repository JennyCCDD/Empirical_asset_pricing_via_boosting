# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191215"

#--* import pakages *--#
import pandas as pd
from sklearn import linear_model
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import pandas_profiling
warnings.filterwarnings("ignore")

#--* import pakages I written*--#
from portfolio_construction import portfolio_EW, portfolio_VW
from performance_measure import performance, performance_anl

#--* define para class*--#
class Para:
    time_window = 12 #months
    rollingtime = 1  #months
    path_data = '.\\normalized\\'
    path = '.\\input\\'
    path_results = '.\\output\\'
    long_short_ratio = 0.1
para = Para()

#--* define multifactor model class*--#
class _factor_:
    def __init__(self):
        return

    def descreptive_an(self,k):
        #--* define function for descreptive_analysis*--#
        for i in range(k - para.time_window, k, para.rollingtime):
            file_name = para.path_data + str(i) + '.csv'
            data_curr_month = pd.read_csv(file_name, header=0, low_memory=False)
            data_curr_month = data_curr_month.dropna(axis=0, how='all')
            profile = data_curr_month.profile_report(title='%s' % k + 'th_month_Exploratory Data Analysis')
            profile.to_file(output_file=para.path_results + '%s' % k + 'th_month_Exploratory Data Analysis.html')
        return

    def data_geter(self,k):
        #--* define function to generate X_in_sample, y_in_sample, X_out_of_sample,y_out_of_sample*--#
        for i in range(k - para.time_window, k, para.rollingtime):
            file_name = para.path_data + str(i) + '.csv'
            data_curr_month = pd.read_csv(file_name, header=0, low_memory=False)
            data_curr_month = data_curr_month.dropna(axis=0, how='all')
            para.n_stock = data_curr_month.shape[0]
            data_in_sample = data_curr_month.fillna(0)
            if k == i - para.time_window:
                data_in_sample = data_curr_month
            else:
                data_in_sample = data_in_sample.append(data_curr_month)
            X_in_sample_ = data_in_sample.loc[:, 'size':'sin']
            X_in_sample_.fillna(X_in_sample_.mean(),inplace = True)
            nom = preprocessing.Normalizer(norm='l2', copy=True)
            X_in_sample_arr = nom.fit_transform(X_in_sample_)
            X_in_sample = pd.DataFrame(X_in_sample_arr,
                                    columns =X_in_sample_.columns,
                                    index = X_in_sample_.index)
            y_in_sample = data_in_sample.loc[:, 'ret']
            y_in_sample.fillna(0,inplace = True)
            file_name2 = para.path_data + str(k) + '.csv'
            data_next_month = pd.read_csv(file_name2, header=0, low_memory=False)
            data_next_month = data_next_month.dropna(axis=0, how='all')
            data_out_of_sample = data_next_month.fillna(0)
            X_out_of_sample_ = data_out_of_sample.loc[:, 'size':'sin']
            X_out_of_sample_.fillna(X_out_of_sample_.mean(),inplace = True)
            nom = preprocessing.Normalizer(norm='l2', copy=True)
            X_out_of_sample_arr = nom.fit_transform(X_out_of_sample_)
            X_out_of_sample = pd.DataFrame(X_out_of_sample_arr,
                                            columns =X_out_of_sample_.columns,
                                            index =  X_out_of_sample_.index)
            y_out_of_sample = data_out_of_sample.loc[:, 'ret']
            y_out_of_sample.fillna(0,inplace = True)
        return X_in_sample, y_in_sample, X_out_of_sample,y_out_of_sample

    def prediction(self,model,X_out_of_sample):
        #--* define function for prediction *--#
        y_score_out_of_sample = model.predict(X_out_of_sample)
        y_score_out_of_sample = pd.DataFrame(y_score_out_of_sample,index = stockid)
        Ret = ret.iloc[:, k - para.time_window]
        RR = rr.iloc[:, k - para.time_window]
        Size = size.iloc[:, k - para.time_window]
        total = pd.concat([y_score_out_of_sample, Ret, Size, RR], axis=1)
        total = total.astype('float')
        total.columns = ['prediction', 'real_return', 'size', 'risk_premium']
        return total

    def model_evaluation(self,k,total):
        #--* define function for model evaluation, especilly for out of sample test *--#
        total.dropna(axis=0, inplace=True)
        MSE = mean_squared_error(total['risk_premium'], total['prediction'])
        r_2 = r2_score(total['risk_premium'], total['prediction'])
        print('test set, month%d, MSE=%.5f' % (k, MSE))
        print('test set, month%d, R^2=%.5f' % (k, r_2))
        return MSE,r_2

if __name__ == '__main__':
    # input: risk premium, monthly return, market size
    rr_input = pd.read_csv(para.path + 'risk_premium_199701.csv', index_col=0, parse_dates=True)
    size_input = pd.read_csv(para.path + 'size_199612.csv', index_col=0, parse_dates=True)
    ret_input = pd.read_csv(para.path + 'final_return_199701.csv', index_col=0, parse_dates=True)
    rf_input = pd.read_csv(para.path+'rf_199701.csv',index_col=0, parse_dates=True)

    # adjust the data to fit out backtest framework
    rr = rr_input.iloc[:,para.time_window:].copy()
    size = size_input.iloc[:,para.time_window:].copy()
    ret = ret_input.iloc[:,para.time_window:].copy()
    rf = rf_input.iloc[:,(para.time_window+1):].copy()

    # get number of month, stockis, datelist
    number_of_month = rr_input.columns.size
    stockid = rr_input.index.tolist()
    datelist = rr.columns.tolist()
    datelist.pop(0)

    # methods in the regression framework
    method_list = ['FM','EXT','GBRT','XGB','LBGM']
    for method in method_list:
        print('__________________%s'% method+'__________________')
        RR_EW = []
        RR_VW = []
        RR_EW_Long = []
        RR_EW_Short = []
        RR_VW_Long = []
        RR_VW_Short = []
        for k in range(int(para.time_window + 1), number_of_month, para.rollingtime):
            #_factor_().descreptive_an(k)
            X_in_sample, y_in_sample, X_out_of_sample,y_out_of_sample = _factor_().data_geter(k)
            if method == 'FM':
                model = linear_model.LinearRegression()#fit_intercept=True, normalize=False, copy_X=True)
            elif method == 'EXT':
                model = ExtraTreesRegressor()
            elif method == 'GBDT':
                model = GradientBoostingRegressor()
            elif method == 'XGB':
                model = xgb.XGBRegressor(silent=True)
            elif method == 'LBGM':
                model = lgb.LGBMRegressor(silent=False)
            else:
                pass
            model.fit(X_in_sample, y_in_sample)
            total =  _factor_().prediction(model,X_out_of_sample)
            _factor_().model_evaluation(k, total)
            total_sort_ew,long_ew, short_ew, long_short_ew = portfolio_EW(total,para.long_short_ratio)
            total_sort_vw,long_vw, short_vw, long_short_vw = portfolio_VW(total,para.long_short_ratio)
            RR_EW_Long.append(long_ew)
            RR_EW_Short.append(short_ew)
            RR_EW.append(long_short_ew)
            RR_VW_Long.append(long_vw)
            RR_VW_Short.append(short_vw)
            RR_VW.append(long_short_vw)

        # main output part
        strategy_RR_EW_Long = pd.DataFrame(RR_EW_Long, columns=['monthly'], index=datelist)
        strategy_RR_EW_Long.index = pd.DatetimeIndex(strategy_RR_EW_Long.index)
        strategy_RR_EW_Long['nav'] = (strategy_RR_EW_Long['monthly'] + 1).cumprod()
        strategy_RR_EW_Long.to_csv(para.path_results + 'strategy_RR_EW_Long_%s' % method + '.csv')
        print('__________________strategy_RR_EW_Long__________________')
        performance(strategy_RR_EW_Long)
        performance_anl(strategy_RR_EW_Long)

        strategy_RR_EW_Short = pd.DataFrame(RR_EW_Short, columns=['monthly'], index=datelist)
        strategy_RR_EW_Short.index = pd.DatetimeIndex(strategy_RR_EW_Short.index)
        strategy_RR_EW_Short['nav'] = (strategy_RR_EW_Short['monthly'] + 1).cumprod()
        strategy_RR_EW_Short.to_csv(para.path_results + 'strategy_RR_EW_Short_%s' % method + '.csv')
        print('__________________strategy_RR_EW_Short__________________')
        performance(strategy_RR_EW_Short)
        performance_anl(strategy_RR_EW_Short)

        strategy_RR_EW = pd.DataFrame(RR_EW, columns=['monthly'], index=datelist)
        strategy_RR_EW.index = pd.DatetimeIndex(strategy_RR_EW.index)
        strategy_RR_EW['nav'] = (strategy_RR_EW['monthly'] + 1).cumprod()
        strategy_RR_EW.to_csv(para.path_results + 'strategy_RR_EW_%s' % method + '.csv')
        print('__________________strategy_RR_EW__________________')
        performance(strategy_RR_EW)
        performance_anl(strategy_RR_EW)

        strategy_RR_VW_Long = pd.DataFrame(RR_VW_Long, columns=['monthly'], index=datelist)
        strategy_RR_VW_Long.index = pd.DatetimeIndex(strategy_RR_VW_Long.index)
        strategy_RR_VW_Long['nav'] = (strategy_RR_VW_Long['monthly'] + 1).cumprod()
        strategy_RR_VW_Long.to_csv(para.path_results + 'strategy_RR_VW_Long_%s' % method + '.csv')
        print('__________________strategy_RR_VW_Long__________________')
        performance(strategy_RR_VW_Long)
        performance_anl(strategy_RR_VW_Long)

        strategy_RR_VW_Short = pd.DataFrame(RR_VW_Short, columns=['monthly'], index=datelist)
        strategy_RR_VW_Short.index = pd.DatetimeIndex(strategy_RR_VW_Short.index)
        strategy_RR_VW_Short['nav'] = (strategy_RR_VW_Short['monthly'] + 1).cumprod()
        strategy_RR_VW_Short.to_csv(para.path_results + 'strategy_RR_VW_Short_%s' % method + '.csv')
        print('__________________strategy_RR_VW_Short__________________')
        performance(strategy_RR_VW_Short)
        performance_anl(strategy_RR_VW_Short)

        strategy_RR_VW = pd.DataFrame(RR_VW, columns=['monthly'], index=datelist)
        strategy_RR_VW.index = pd.DatetimeIndex(strategy_RR_VW.index)
        strategy_RR_VW['nav'] = (strategy_RR_VW['monthly'] + 1).cumprod()
        strategy_RR_VW.to_csv(para.path_results + 'strategy_RR_VW_%s' % method + '.csv')
        print('__________________strategy_RR_VW__________________')
        performance(strategy_RR_VW)
        performance_anl(strategy_RR_VW)

"""
I wanna put them into a dataframe, but failed
to do this, I need to redefine the perfermance and perfoermance_anl functions
but I failed about how to get the key word from the columns to make all the portfolios suited for the same function
I do not wanna to use the [1],[2]

def performance(strategy):
    def MaxDrawdown(return_list):
        RET_ACC = []
        sum = 1
        for i in range(len(return_list)):
            sum = sum * (return_list[i] + 1)
            RET_ACC.append(sum)
        index_j = np.argmax(np.array((np.maximum.accumulate(RET_ACC) - RET_ACC) / np.maximum.accumulate(RET_ACC)))
        index_i = np.argmax(RET_ACC[:index_j])
        MDD = (RET_ACC[index_i] - RET_ACC[index_j]) / RET_ACC[index_i]
        return sum, MDD, RET_ACC

    def sharp(return_list, std):
        #returnew = pd.DataFrame(return_list, columns=['R'])
        #m = pd.concat([returnew.R,rf], axis=1)
        ret_adj = np.array(return_list)
        sharpratio = np.average(ret_adj) * 12 / std
        return sharpratio

    def Reward_to_VaR(strategy=strategy, alpha=0.99):
        RET = strategy['monthly']
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        var = abs(sorted_Returns[index])
        RtoVaR = np.average(RET) / var
        return -RtoVaR

    def Reward_to_CVaR(strategy=strategy, alpha=0.99):
        RET = strategy['monthly']
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        sum_var = sorted_Returns[0]
        for i in range(1, index):
            sum_var += sorted_Returns[i]
            CVaR = abs(sum_var / index)
        RtoCVaR = np.average(RET) / CVaR
        return -RtoCVaR

    strategy_header = [ strategy.columns.tolist().astype(str).split('_')]
    strategy = strategy[strategy_header:].copy()
    ts = strategy['monthly']

    ts_list = ts.tolist()
    #RET =ts.mean()*12
    RET = (strategy.nav[strategy.shape[0] - 1] /strategy.nav[0])** (12 / strategy.shape[0]) - 1
    T = stats.ttest_1samp(ts, 0)[0]
    STD = np.std(ts) * np.sqrt(252)
    MDD = MaxDrawdown(ts_list)[1]
    ACC = MaxDrawdown(ts_list)[0]
    SHARP = (RET-0.03)/ STD
    #SHARP = sharp(ts,STD)
    R2VaR = Reward_to_VaR(strategy)
    R2CVaR = Reward_to_CVaR(strategy)
    print('annual-return', round(RET, 4))
    print('t-statistic', round(T, 4))
    print('volitility', round(STD, 4))
    print('MaxDrawdown', round(MDD, 4))
    print('Accumulated return', round(ACC, 4))
    print('sharp-ratio', round(SHARP, 4))
    print('Reward_to_VaR', round(R2VaR, 4))
    print('Reward_to_CVaR', round(R2CVaR, 4))
    return RET, T, STD, MDD, ACC, SHARP, R2VaR, R2CVaR
    
#________here is the codes to generate the dataframe of strategy:
    
        strategy  = pd.DataFrame(RR_EW_Long,columns = ['monthly_RR_EW_Long'],index=datelist)
        strategy.index = pd.DatetimeIndex(strategy.index)
        strategy['nav_RR_EW_Long'] = (strategy['monthly_RR_EW_Long'] +1).cumprod()
        strategy_RR_EW_Long = strategy.loc[:,'monthly_RR_EW_Long':'nav_RR_EW_Long']
        print('__________________strategy_RR_EW_Long__________________')
        performance(strategy_RR_EW_Long)
        performance_anl(strategy_RR_EW_Long)

        strategy['monthly_RR_EW_Short']  = RR_EW_Short
        #strategy.index = pd.DatetimeIndex(strategy.index)
        strategy['nav_RR_EW_Short'] = (strategy['monthly_RR_EW_Short'] +1).cumprod()
        strategy_RR_EW_Short = strategy.loc[:,'monthly_RR_EW_Short':'monthly_RR_EW_Short']
        print('__________________strategy_RR_EW_Short__________________')
        performance(strategy_RR_EW_Short)
        performance_anl(strategy_RR_EW_Short)

        strategy['monthly_RR_EW'] = RR_EW
        #strategy.index = pd.DatetimeIndex(strategy_RR_EW.index)
        strategy['nav_RR_EW'] = (strategy['monthly_RR_EW'] + 1).cumprod()
        strategy_RR_EW = strategy.loc[:,'monthly_RR_EW':'nav_RR_EW']
        print('__________________strategy_RR_EW__________________')
        performance(strategy_RR_EW)
        performance_anl(strategy_RR_EW)

        strategy['monthly_RR_VW_Long']= RR_VW_Long
        #strategy.index = pd.DatetimeIndex(strategy.index)
        strategy['nav_RR_VW_Long'] = (strategy['monthly_RR_VW_Long'] +1).cumprod()
        strategy_RR_VW_Long = strategy.loc[:,'monthly_RR_VW_Long':'nav_RR_VW_Long']
        print('__________________strategy_RR_VW_Long__________________')
        performance(strategy_RR_VW_Long)
        performance_anl(strategy_RR_VW_Long)

        strategy['monthly_RR_VW_Short']  = RR_VW_Short
        #strategy.index = pd.DatetimeIndex(strategy.index)
        strategy['nav_RR_VW_Short'] = (strategy['monthly_RR_VW_Short'] +1).cumprod()
        strategy_RR_EW_Short = strategy.loc[:,'monthly_RR_EW_Short':'monthly_RR_EW_Short']
        print('__________________strategy_RR_EW_Short__________________')
        performance(strategy_RR_EW_Short)
        performance_anl(strategy_RR_EW_Short)

        strategy['monthly_RR_EW'] =RR_EW
        #strategy.index = pd.DatetimeIndex(strategy_RR_EW.index)
        strategy['nav_RR_EW'] = (strategy['monthly_RR_EW'] + 1).cumprod()
        strategy_RR_EW = strategy.loc[:,'monthly_RR_EW':'nav_RR_EW']
        print('__________________strategy_RR_EW__________________')
        performance(strategy_RR_EW)
        performance_anl(strategy_RR_EW)

        strategy.to_csv(para.path_results+'strategy_'%method+'.csv')
        """




