# timeseries

Time-series analysis is adventurous I'd say, forecasting with time-series data can be challenging. 
<img width="500" alt="image" src="https://github.com/ranja-sarkar/Time_series/assets/101544669/aa888439-8a6e-4088-beae-11fc7dd6146f">


Some points to keep in mind while dealing with time-series:

1) The uncertainty of forecast is just as important as the (point) forecast itself.
2) Model serving (deploying & scoring) is really tricky.
3) Cross-validation (with sliding or expanding window as testing strategy) is also tricky.
<img width="491" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/0dcf3f84-cc5c-4b16-98a5-58c4f2cddc9e">

In general, some time-series exhibit ill-behaved uncertainty. The forecast errors do not follow known distributions. Such information is useful for making judgmental decisions, but cannot be modeled and used for forecasting. Such an uncertainty is coconut uncertainty - of unknown unknowns leading to unpredictability. 

<img width="334" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/05b48529-29cc-4166-95ee-795a20961b5a">


Other time-series exhibit well-behaved uncertainty. The forecast errors follow known distributions - Normal, Poisson etc.. Such information is useful for modeling and predictions bound within a certain range. This window of uncertainty is subway uncertainty - of known unknowns. 
<img width="341" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/a6c5417f-50e5-4bd8-aefc-18bab5809f8a">

Forecast of level + trend is a baseline forecast. Baseline forecasts with the persistence model (Using an observation at the previous time step to learn what will happen in the next time step)
quickly indicate whether you can do significantly better. If you can’t, you’re probably dealing with a random walk. 
The human mind is hardwired to look for patterns everywhere and we must be vigilant we're not fooling ourselves and wasting time by developing elaborate models for random walk processes.

**Approaches to smoothing a time-series: Baseline models**

Holt's method - there're level smoothing constant (alpha) and trend constant (beta).

Holt Winter's method - there's seasonal smoothing constant (delta) and considers seasonal baseline which is a regularly recurring pattern (day, week, month, quarter etc.) and baseline rises and falls at regular intervals. Deviation of each season from the baseline’s long-term (annual) average is used for forecasts. 

Exponential Smoothing - Defines trend as the difference between observed values in consecutive (in time) records.

Smoothing models are for removal of noise. Moving averages are considered for these and they can be simple, exponential, and cumulative.
Examples: https://www.kaggle.com/code/ranja7/energy-consumption-forecast-baseline-models

**Forecasting approach**

ARIMA handles data with trend.
SARIMA handles data with a seasonal component. 
The trend, seasonality and noise in a time series are explained by model parameter set (p,d,q), also called the order. The auto-regressive parameter is p; d is difference parameter and q is the moving average parameter. Trend is the long-term change in the mean level of observations. Seasonality is the pattern that’s periodically repeated, and noise is the random variation in the data. A time series is additive when the 'trend' is linear (changes are at linear rate) and 'seasonality' is constant in time. A time series is multiplicative when the 'trend' is nonl-linear. A stationary time series has constant mean and variance and does not exhibit a trend.

Y(t) = Level + Trend + Seasonality + Noise

Example: https://www.kaggle.com/code/ranja7/sarima-forecasts-auto-arima

For multivariate time-series data, one can follow VAR (vector autoregression) approach. One can utilize deep learning methods like LSTM in multivariate time-series.
https://www.kaggle.com/code/ranja7/forecasting-with-lstm-tensorflow

**Other libraries used for forecasting**

https://unit8.com/resources/darts-time-series-made-easy-in-python/
<img width="279" alt="image" src="https://github.com/ranja-sarkar/Time_series/assets/101544669/2702a5df-5000-4fb9-829c-dd1a1f6abdb7">

https://pypi.org/project/statsforecast/

There can be outliers in time-series data, often called anomalies due to their deviation from 'normal'. Anomalies can be point or collective (subsequent). 

https://www.kaggle.com/code/ranja7/anomaly-detection-in-timeseries-isolation-forest


One can also use PROPHET for anomaly detection and forecasting:
https://facebook.github.io/prophet/docs/outliers.html




