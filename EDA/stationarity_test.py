
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

