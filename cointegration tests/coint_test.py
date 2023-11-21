
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank
from statsmodels.tsa.stattools import grangercausalitytests

df = pd.read_csv('../input/power-factor-timeseries/PowerFactor.csv')
df0 = df.dropna(how = 'any', axis = 0) #drop empty rows if any
df0.info(verbose = False)

df1 = pd.read_csv('../input/power-factor-timeseries/CurrentVoltage.csv')
df2 = df1.dropna(how = 'any', axis = 0) #drop empty rows if any
df2.info()

def resampled_timeseries(filename):
    '''
    Resampling of input data (down-sampling) -> high to low frequency
    '''
    df = pd.read_csv(filename, parse_dates = ['DeviceTimeStamp'])
    df = df.dropna(how = 'any', axis = 0)
    df.set_index('DeviceTimeStamp', inplace = True)
    df['Date'] = df.index
    
    dff = pd.DataFrame(columns = ['PFL1', 'THDVL1', 'THDIL1', 'MDIL1'])
    new_df = pd.DataFrame(columns = ['PFL1', 'THDVL1', 'THDIL1', 'MDIL1'])
    dff['PFL1'] = df['PFL1'].resample('H').mean().ffill()
    dff['THDVL1'] = df['THDVL1'].resample('H').mean().ffill()
    dff['THDIL1'] = df['THDIL1'].resample('H').mean().ffill()
    dff['MDIL1'] = df['MDIL1'].resample('H').mean().ffill()
    new_df = new_df.append(dff)
    new_df['Date'] = new_df.index

    new_df.to_csv('Resampled_PF.csv', index = False)
    return print('Resampled file created.')

def resampled_series(filename):
    '''
    Resampling of input data (down-sampling) -> high to low frequency
    '''
    df = pd.read_csv(filename, parse_dates = ['DeviceTimeStamp'])
    df = df.dropna(how = 'any', axis = 0)
    df.set_index('DeviceTimeStamp', inplace = True)
    df['Date'] = df.index
    
    dff = pd.DataFrame(columns = ['VL1', 'IL1', 'VL12', 'VL31', 'INUT'])
    new_df = pd.DataFrame(columns = ['VL1', 'IL1', 'VL12', 'VL31', 'INUT'])
    dff['VL1'] = df['VL1'].resample('H').mean().ffill()
    dff['IL1'] = df['IL1'].resample('H').mean().ffill()
    dff['VL12'] = df['VL12'].resample('H').mean().ffill()
    dff['VL31'] = df['VL31'].resample('H').mean().ffill()
    dff['INUT'] = df['INUT'].resample('H').mean().ffill()
    new_df = new_df.append(dff)
    new_df['Date'] = new_df.index

    new_df.to_csv('Resampled_VC.csv', index = False)
    return print('Resampled file created.')

dff1 = pd.read_csv('./Resampled_PF.csv')
#dff1.info()
dff2 = pd.read_csv('./Resampled_VC.csv')
#dff2.info()

nobs = 720 #timseteps #30days (30*24hours)
df_train, df_test = dff1[0:-nobs], dff1[-nobs:]
tt1 = round(((len(df_train)/len(dff1))*100), 1)
print("Train % = ", tt1)
tt2 = round(((len(df_test)/len(dff1))*100), 1)
print("Test % = ", tt2)
print("Mean of variables: ")
print(round(np.mean(df_train), 2))

n = len(df_train)
train = df_train[['PFL1', 'THDVL1', 'THDIL1', 'MDIL1']].loc[0: n-1]

y1 = df_train['PFL1']
plot_acf(y1) 
plot_pacf(y1) 
plt.title('PFL1 - PACF')
plt.show()

nobs = 720 
df1_train, df1_test = dff2[0:-nobs], dff2[-nobs:]
tt1 = round(((len(df1_train)/len(dff2))*100), 1)
print("Train % = ", tt1)
tt2 = round(((len(df1_test)/len(dff2))*100), 1)
print("Test % = ", tt2)
print("Mean of variables: ")
print(round(np.mean(df1_train), 2))

n1 = len(df1_train)
train1 = df1_train[['VL1', 'IL1', 'VL12', 'VL31', 'INUT']].loc[0: n1-1]

y11 = df1_train['INUT']
plot_acf(y11) 
plot_pacf(y11) 
#plt.title('PACF')
plt.show()

#JOHANSEN'S TESTS

#coint_johansen(train,1, 1).eig  
##1 for linear trend ##-1 for non-deterministic trend ##1 for #lags

vec_rank1 = select_coint_rank(df_train[['PFL1','THDVL1', 'THDIL1', 'MDIL1']], det_order = 1, k_ar_diff = 1, method = 'maxeig', signif = 0.05)
print(vec_rank1.summary())

vec_rank2 = select_coint_rank(df_train[['PFL1','THDVL1', 'THDIL1', 'MDIL1']], det_order = 1, k_ar_diff = 1, method = 'trace', signif = 0.05)
print(vec_rank2.summary())

vec_rank3 = select_coint_rank(df1_train[['VL1', 'IL1', 'VL12', 'VL31', 'INUT']], det_order = 1, k_ar_diff = 1, method = 'maxeig', signif = 0.05)
print(vec_rank3.summary())

vec_rank4 = select_coint_rank(df1_train[['VL1', 'IL1', 'VL12', 'VL31', 'INUT']], det_order = 1, k_ar_diff = 1, method = 'trace', signif = 0.05)
print(vec_rank4.summary())

#When two or more time series are cointegrated, it means they have a statistically significant (long run) relationship.
#There exists a linear combination of them that has an order of integration (differencing required to make a non-stationary time series stationary) less than that of the individual series, and the collection of series is said to be cointegrated.

def cointegration_test(df, alpha = 0.05): 
    """Perform Johansen's Cointegration Test, report summary"""
    
    out = coint_johansen(df,1,1)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)
        
    print('Name   ::  Test Stat. > C(95%)    =>   Significance  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df_train[['PFL1','THDVL1', 'THDIL1', 'MDIL1']])
cointegration_test(df1_train[['VL1', 'IL1', 'VL12', 'VL31', 'INUT']])

#GRANGER'S CAUSALITY TEST
#It tests the null hypothesis that the coefficients of past values in the regression equation is zero. 
#In simpler terms, the past values of time series (X) do not cause the other series (Y). 
#If the p-value obtained from the test is lesser than the significance level (0.05), you can safely reject the null hypothesis.

def grangers_causation_mat(data, variables, test = 'ssr_chi2test', verbose = False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the p-values. p-values lesser than 0.05, implies the Null Hypothesis (coefficients of the 
    corresponding past values is zero), meaning X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns = variables, index = variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag = maxlag, verbose = False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
            
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

maxlag = 3
var = df_train[['PFL1','THDVL1', 'THDIL1', 'MDIL1']].columns
grangers_causation_mat(df_train[['PFL1','THDVL1', 'THDIL1', 'MDIL1']], variables = var)   

maxlag = 1
var1 = df1_train[['VL1', 'IL1', 'VL12', 'VL31', 'INUT']].columns
grangers_causation_mat(df1_train[['VL1', 'IL1', 'VL12', 'VL31', 'INUT']], variables = var1) 

#If a given p-value is less than 0.05, the corresponding X series causes Y.
#Looking at the p-values, all the variables in the system are interchangeably causing each other.
