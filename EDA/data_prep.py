#Data prep for LSTM

import pandas as pd
from pandas import concat
 
def timeseries_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    """
    Arguments:
        data: Sequence of observations as a list or NumPy array
        n_in: Number of lag observations as input
        n_out: Number of observations as output
        dropnan: whether or not to drop rows with NaN values 
    Returns:
        Pandas DataFrame for supervised learning.
    """
    
    n_vars = 1 if type(data) is list else data.shape[1]
    
    df = pd.DataFrame(data)
    cols, names = list(), list()

    #input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = concat(cols, axis = 1)
    agg.columns = names

    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace = True)
    return agg

#Ex1
values = [x for x in range(10)]
data = timeseries_to_supervised(values) #lag timestep = 1
print(data)

#Ex2
values = [x for x in range(10)]
data = timeseries_to_supervised(values, 3)
print(data)

#Ex3
values = [x for x in range(10)]
data = timeseries_to_supervised(values, 1, 2)
print(data)

#Ex4
raw = pd.DataFrame()
raw['obs1'] = [x for x in range(10)]
raw['obs2'] = [x for x in range(50, 60)]
values = raw.values
data = timeseries_to_supervised(values)
print(data)

#Ex5
raw = pd.DataFrame()
raw['obs1'] = [x for x in range(10)]
raw['obs2'] = [x for x in range(50, 60)]
values = raw.values
data = timeseries_to_supervised(values, 1, 2)
print(data)
