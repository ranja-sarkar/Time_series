
#ADF Test
def adf_test(data):
  """ Checks stationarity of timeseries """
  adf = adfuller(data)
  output = pd.Series(adf[0:3], index = ['ADF Statistic', 'p-value', 'Lags'])
  for key, value in adf[4].items():
    output["Critical Value (%s)" %key] = value
  return output

#KPSS Test
def kpss_test(dataseries):
    """Checks stationarity of timeseries"""
  kpss_input = kpss(data)
  output = pd.Series(kpss_input[0:3], index = ['KPSS Statistic', 'p-value', 'Lags'])
  for key, value in kpss_input[3].items():
    output["Critical Value (%s)" %key] = value
  return output

#print(adf_test(train['']))
#print('\n \n')
#print(kpss_test(train['']))

#adf_test(train.diff().dropna())
#kpss_test(train.diff().dropna())

##KPSS assumes the series is stationary, non-stationary if p-value<0.05 or if test statistic is above critical value


