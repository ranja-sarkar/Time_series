#exploring time-series components

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('Sales.csv')
df.info()

#df['Date'] = pd.to_datetime(df['Date(mm-dd-yy)'])
df.set_index('Date(mm-dd-yy)', inplace = True)
df['Date'] = df.index
df.describe().T

#df['Date'].describe()

pd.plotting.lag_plot(df['Quantity'])
#plt.grid()

plt.figure(figsize = (8, 6))
df['Quantity'].cumsum().plot()
plt.ylabel('Quantity')

pd.plotting.autocorrelation_plot(df['Quantity'])
plt.show()

shift1 = df['Quantity'].autocorr()
print('Autocorrelation = ', shift1.round(4))

plot_acf(df['Trucks Sold'])
plt.xlabel('Lag')
plt.grid()

plot_pacf(df['Quantity']) 
plt.xlabel('Lag')
plt.grid()
plt.show()

#default additive model
result1 = seasonal_decompose(df['Trucks Sold'].loc['5/13/2014':], model = 'additive', period = 1)
fig = result.plot()
plt.xticks(rotation = 90)
plt.xlabel('Date')
fig.set_size_inches(20, 10)

result2 = seasonal_decompose(df['Trucks Sold'].loc['5/13/2014':], model = 'multiplicative', period = 1)
fig = result.plot()
plt.xticks(rotation = 90)
plt.xlabel('Date')
fig.set_size_inches(20, 10)

plt.figure(figsize = (20, 10))
plt.ylabel('Quantity')
plt.xlabel('Date')
plt.plot(df['Trucks Sold'])
plt.xticks(rotation = 90)
plt.grid()



