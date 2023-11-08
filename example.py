#Forcasting using DARTS

import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import matplotlib.pyplot as plt


train, test = series.split_before(pd.Timestamp('2020-10-15'))

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))

series.plot(label = 'actual')
prediction.plot(label = 'forecast', lw = 1.5)
plt.title('Energy Consumption (kWh)')
plt.legend()

