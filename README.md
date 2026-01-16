# Forecasting

When information is transferred across time, often to discrete points in time, it is called forecasting. Forecast of the future using temporal behavior of data need uncertainty quantification, so we stay aware of the level of our confidence in the forecasts or point estimates.

A time series is a set of data points (indexed in time), and is most commonly a sequence of successive equally spaced out points in time. 

Time-series data can be continuous → ordinary differential equations (ODEs) and stochastic differential equations (SDEs) are continuous-time models. In the figure below, the horizon or period of forecast is 3 days and there'e 80% confidence in the forecasts. 

![tt](https://github.com/user-attachments/assets/45d93530-4c6e-4819-b430-d05d175aaa5a)

If we were to say that the timestamp is our feature and the value at that time is our dependent variable, we have a regression problem. The observed pattern in time-series data is the level - it’s the mean value over a specific period. The behavioral patterns in the data yield what are called components of time-series. 

# Time-series components

1. **Trend** → Persistent, long-term behavior (linear or non-linear)

2. **Seasonality** → Regular, periodic behavior within a year. A Fourier transform of time-series helps detect seasonality in the data, basically identifies the frequency peaks and corresponding amplitudes of the time-series signal. 

3. **Cyclicity** → Repeated or recurring behavior over more than a year

4. **Residual** → Erratic or irregular behavior (noise). The randomness of residual makes it a ‘white noise’ (contains all frequencies). The white noise does not influence the mean value of time-series.

![patt](https://github.com/user-attachments/assets/e6fb229b-beff-412c-acbb-bc3c82434e0f)

Then there is **autocorrelation** - value at a certain point in time depends on prior values. An autoregressive (AR) characteristic of time-series, the sequential dependency of observations usually introduces structured errors that are correlated temporally on previous errors or observation(s).

The autocorrelation, also called the AR component models changes in the time-series that are not explained by trend or seasonality.  

# Forecasting Approaches

**ARIMA** (auto-regressive integrated moving average) handles data with trend. **SARIMA** handles data with a seasonal component. 

The trend, seasonality and noise in a time series are explained by model parameter set (p,d,q), also called the order. The auto-regressive parameter is p; d is difference parameter and q is the moving average parameter. Trend is the long-term change in the mean level of observations. Seasonality is the pattern that’s periodically repeated, and noise is the random variation in the data. 

A time series is additive when the 'trend' is linear (changes in mean and variance are at linear rate) and 'seasonality' is constant in time. A time series is multiplicative when the 'trend' is non-linear. A stationary time series has constant mean over time and does not exhibit a trend. Having no trend means identically distributed random variables.  

Y(t) = Level + Trend + Seasonality + Noise 

📌 A stationary time-series does not exhibit a trend. 

Stationarity means that the mean and variance are constant and do not change over time. The trend is how the mean value of a time-series changes over time, the seasonality is how the mean value of a time-series changes over seasons in a year. 

For a non-stationary time series, there are two approaches to model the trend and seasonality - detrending, differencing.

In detrending, use a regression model, slope of the linear regression fit is the increase/decrease of y-value with x (time). Detrending only works for linear trends. 

<img width="568" height="190" alt="ts1" src="https://github.com/user-attachments/assets/f85b9ea2-79ec-42cb-ae8c-f5bc197e530a" />

In the figure above showing ACF and PACF, there’s a strong residual seasonal effect upon detrending. The time-series has not become fully stationary.

In differencing, use difference between two consecutive values in time. Differencing can deal with both changes in the mean (first-order difference) as well as changes in the mean movement (second-order difference). 

<img width="556" height="188" alt="ts2" src="https://github.com/user-attachments/assets/eb60b373-e762-4520-b5aa-58ce884e4ab8" />

In the figure above, there is no correlation between the time steps. Stationarity has been achieved by differencing.

📌 Partial autocorrelation is the autocorrelation without the carry-over (correlation between non-consecutive points in the time-series).

---


Heteroscedasticity happens when the standard errors of a variable, monitored over a specific amount of time are non-constant. Conditional heteroscedasticity identifies non-constant volatility (degree of variation of series over time) when future periods of high & low volatility cannot be identified. Unconditional heteroscedasticity is used when future high & low volatility periods can be identified.

Some points to keep in mind while dealing with time-series data:

1) The uncertainty of forecast is just as important as the (point) forecast itself.
   
2) Model serving (deploying & scoring) is challenging.

3) Cross-validation (with sliding or expanding window as testing strategy) is tricky.
   
<img width="491" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/0dcf3f84-cc5c-4b16-98a5-58c4f2cddc9e">

Find the link to **time-series backtesting strategies** with the python library **skforecast** below. 

<img width="298" alt="00" src="https://github.com/user-attachments/assets/b5c4fa17-105c-46f9-9706-6586d19eae20">

# Confidence in forecasts

There are several tools for quantification of uncertainty (in forecasts), each with their drawbacks like not having guaranteed coverage, conformal prediction however fills the gap. 

![cp](https://github.com/user-attachments/assets/eadf0b62-3f0c-4a92-abfe-e37c06bfdd12)


Some time-series exhibit ill-behaved uncertainty. The forecast errors do not follow known distributions. Such information is useful for making judgmental decisions, but cannot be modeled and used for forecasting. Such an uncertainty is coconut uncertainty - of unknown unknowns leading to unpredictability. 

<img width="334" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/05b48529-29cc-4166-95ee-795a20961b5a">


Other time-series exhibit well-behaved uncertainty. The forecast errors follow known distributions - normal, Poisson etc.. Such information is useful for modeling and predictions bound within a certain range. The window of uncertainty is subway uncertainty - of known unknowns. 

<img width="341" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/a6c5417f-50e5-4bd8-aefc-18bab5809f8a">


Subway uncertainty is the uncertainty of known unknowns; events can be predicted assuming a window of uncertainty. It is like, even on a worst day trying to navigate through rush hour trains, your subway voyage almost definitely won't take you more than around 30 minutes plus planned time. 

Coconut uncertainty is an allusion to a coconut unexpectedly falling on one’s head while on a beach. It is the uncertainty of unknown unknowns due to ill-behaved forecast errors; events that could never be predicted no matter how hard we tried.

# Baseline Models

Forecast of level + trend is a baseline forecast. Baseline forecasts with the persistence model (using an observation at the previous time step to learn what will happen in the next time step) quickly indicate whether you can do significantly better. Baseline forecasts quickly indicate whether you can have significantly better forecasts by utilizing advanced models. If there’s no recognizable pattern, you’re probably dealing with a random walk process (future is independent of past). 

The human mind is hardwired to look for patterns everywhere and we must be vigilant if we're developing elaborate models for random walk processes. A [random walk](https://en.wikipedia.org/wiki/Random_walk) having a step size that varies according to a normal distribution is used as a model for real-world time-series data such as financial markets.

Approaches to smoothing a time-series:

1. **Holt's method** - there're level smoothing constant (alpha) and trend constant (beta).

2. **Holt Winter's method** - there's seasonal smoothing constant (delta) and considers seasonal baseline which is a regularly recurring pattern (day, week, month, quarter etc.) and baseline rises and falls at regular intervals. Deviation of each season from the baseline’s long-term (annual) average is used for forecasts. 

Smoothing models are for removal of noise. 

3. **Moving averages** are smoothing models, they can be **simple**, **exponential**, and **cumulative**. Moving average models are used to model the effect of random fluctuations in the time-series. We can say the influence of random fluctuations is relatively small when the autocorrelation goes toward zero. 


Simple Moving Average (SMA) uses a sliding window to take the average over a set number of time periods. It is an equally weighted mean of the previous data in this period.
Unlike SMA which drops the oldest observation as the new one gets added, cumulative moving average (CMA) considers all prior observations. 
Unlike SMA and CMA, exponential moving average (EMA) gives more weight to the recent prices as a result of which, EMA can better capture the movement of the trend. Exponential smoothing defines trend as the difference between observed values in consecutive records in time.

# Forecasting Libraries

**Darts** is a python library by **Unit8** for forecasting. 

Darts: https://unit8.com/resources/darts-time-series-made-easy-in-python/

<img width="279" alt="image" src="https://github.com/ranja-sarkar/Time_series/assets/101544669/2702a5df-5000-4fb9-829c-dd1a1f6abdb7">

**NIXTLA libraries for forecasting**: Nixtla democratizes access to SOTA.

statsforecast: https://pypi.org/project/statsforecast/

statsforecast repo: https://github.com/Nixtla/statsforecast

Open-source time-series ecosystem by **NIXTLA**: Fast and easy-to-use tools for forecasting & anomaly detection

<img width="554" alt="0" src="https://github.com/user-attachments/assets/799983af-3d0f-4f4e-96d6-082f2891ef99">


📌 [sktime](https://www.sktime.net/en/latest/)

📌 [skforecast](https://skforecast.org/0.14.0/index.html)

skforecast has [backtesting strategies](https://skforecast.org/0.14.0/user_guides/backtesting.html).

📌 [tslearn](https://github.com/tslearn-team/tslearn)

📌 [autoTS](https://github.com/winedarksea/AutoTS)

Please note 'autoTS' is different from [auto_ts](https://colab.research.google.com/drive/1OgvVA1XLRle3gUBtu2tVcHmSTqR1-vP3).


Outliers or anomalies in time-series data can be point or collective (subsequent). 


📌 [PROPHET (by Meta)](https://facebook.github.io/prophet/docs/outliers.html)


📌 [puncc](https://github.com/deel-ai/puncc) for [conformal prediction](https://colab.research.google.com/drive/1iP36nrvTge18kdNZPOXjn5nje8SMj-bk)


📌 [merlion (by Salesforce)](https://github.com/salesforce/Merlion)

<img width="503" alt="1" src="https://github.com/user-attachments/assets/e076da9d-425e-4a21-b547-abe10982e484">


# Multivariate time-series

For multivariate time-series data, one can follow VAR or [vector autoregression](https://gist.github.com/ranja-sarkar/6b79ae244ef31a6034324a3ca0de8c10) approach. 

One can develop dynamic models to investigate short-term and long-term causal effects. Short-term causality might be observed through Granger causality, while long-term causality is typically studied using techniques like ECM or VECM. 

<img width="598" height="326" alt="latt" src="https://github.com/user-attachments/assets/85f89afc-d150-4e29-b51c-df5a5a7f1780" />


One can utilize deep learning algorithms like LSTM (with either [tensorflow](https://github.com/ranja-sarkar/ranja-sarkar.github.io/blob/23ac98dc7fc119547fb094757165395fe061fff4/_posts/codes/LSTM_with_Tensorflow.ipynb) or pytorch) in multivariate time-series.


# Probabilistic modeling with time-series

For probabilistic time-series modeling, one can use AutoGluon's [gluonts](https://github.com/awslabs/gluonts) from AWS labs.

[Chronos](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html) is a family of pretrained time series forecasting models. Chronos models are based on language model architectures, and work by quantizing time-series into buckets which are treated as tokens.

[Here's](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting) how to adapt to LM architectures for forecasting from Amazon Science!!

And there's also [Bayesian forecasting](https://github.com/ChadFulton/scipy2022-bayesian-time-series) using statsmodels.


# Foundation Model for forecasting

[TimeGPT](https://github.com/Nixtla/nixtla) is production ready pre-trained time-series foundation model for forecasting & anomaly detection by [NIXTLA](https://www.nixtla.io/), which democratizes access to SOTA open-source tools as well as enterprise APIs for anomaly detection and forecasting.







