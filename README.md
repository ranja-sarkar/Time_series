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

**ARIMA** handles data with trend.
**SARIMA** handles data with a seasonal component. 
The trend, seasonality and noise in a time series are explained by model parameter set (p,d,q), also called the order. The auto-regressive parameter is p; d is difference parameter and q is the moving average parameter. Trend is the long-term change in the mean level of observations. Seasonality is the pattern that’s periodically repeated, and noise is the random variation in the data. 

A time series is additive when the 'trend' is linear (changes in mean and variance are at linear rate) and 'seasonality' is constant in time. A time series is multiplicative when the 'trend' is non-linear. A stationary time series has constant mean over time and does not exhibit a trend. Having no trend means identically distributed random variables.  

Y(t) = Level + Trend + Seasonality + Noise 

---

Heteroscedasticity happens when the standard errors of a variable, monitored over a specific amount of time are non-constant. Conditional heteroscedasticity identifies non-constant volatility (degree of variation of series over time) when future periods of high & low volatility cannot be identified. Unconditional heteroscedasticity is used when future high & low volatility periods can be identified.

Some points to keep in mind while dealing with time-series data:

1) The uncertainty of forecast is just as important as the (point) forecast itself.
2) Model serving (deploying & scoring) is challenging.
3) Cross-validation (with sliding or expanding window as testing strategy) is tricky.
   
<img width="491" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/0dcf3f84-cc5c-4b16-98a5-58c4f2cddc9e">

Find the link to **time-series backtesting strategies** with the python library **skforecast** below. 

<img width="298" alt="00" src="https://github.com/user-attachments/assets/b5c4fa17-105c-46f9-9706-6586d19eae20">


In general, some time-series exhibit ill-behaved uncertainty. The forecast errors do not follow known distributions. Such information is useful for making judgmental decisions, but cannot be modeled and used for forecasting. Such an uncertainty is coconut uncertainty - of unknown unknowns leading to unpredictability. 

<img width="334" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/05b48529-29cc-4166-95ee-795a20961b5a">


Other time-series exhibit well-behaved uncertainty. The forecast errors follow known distributions - Normal, Poisson etc.. Such information is useful for modeling and predictions bound within a certain range. This window of uncertainty is subway uncertainty - of known unknowns. 

<img width="341" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/a6c5417f-50e5-4bd8-aefc-18bab5809f8a">

Forecast of level + trend is a baseline forecast. Baseline forecasts with the persistence model (using an observation at the previous time step to learn what will happen in the next time step)
quickly indicate whether you can do significantly better. 
The human mind is hardwired to look for patterns everywhere and we must be vigilant if we're developing elaborate models for random walk processes. A random walk having a step size that varies according to a normal distribution is used as a model for real-world time series data such as financial markets.

https://en.wikipedia.org/wiki/Random_walk


**Baseline Models:** Approaches to smoothing a time-series

1. **Holt's method** - there're level smoothing constant (alpha) and trend constant (beta).

2. **Holt Winter's method** - there's seasonal smoothing constant (delta) and considers seasonal baseline which is a regularly recurring pattern (day, week, month, quarter etc.) and baseline rises and falls at regular intervals. Deviation of each season from the baseline’s long-term (annual) average is used for forecasts. 

Smoothing models are for removal of noise. 

3. **Moving averages** are smoothing models, they can be **simple**, **exponential**, and **cumulative**.

Example code: https://colab.research.google.com/drive/1AwqNPjCbh7kXc1GsmYLOXJU8aC2Pa-vE

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

Projects repo: https://github.com/Nixtla/

**Generative AI for time-series by NIXTLA**: TimeGPT

<img width="881" alt="1" src="https://github.com/user-attachments/assets/58194ee9-aeef-4b72-b647-f4fdb4dfcafe">


**Other libraries for forecasting**

sktime: https://www.sktime.net/en/latest/

📌 skforecast: https://skforecast.org/0.14.0/index.html

📌 Backtesting strategies with skforecast: https://skforecast.org/0.14.0/user_guides/backtesting.html

tslearn: https://github.com/tslearn-team/tslearn

autoTS: https://github.com/winedarksea/AutoTS


Please note this 'autoTS' is different from 'auto_ts' used here:
https://colab.research.google.com/drive/1OgvVA1XLRle3gUBtu2tVcHmSTqR1-vP3


Outliers  (deviations from 'normal') in time-series data, also called anomalies can be point or collective (subsequent). 

https://colab.research.google.com/drive/1GNEkNqJp9sb8BrxOzY0uYK66OxtjCmE1


One can also use **PROPHET (by Meta)** for anomaly detection and forecasting:
https://facebook.github.io/prophet/docs/outliers.html

📌📌 **Conformal prediction over time**

Conformal anomaly detection in timeseries (demo) example: https://colab.research.google.com/drive/1iP36nrvTge18kdNZPOXjn5nje8SMj-bk

📌 Conformal Prediction: https://github.com/deel-ai/puncc

Conformal Regression & Classification:  https://github.com/deel-ai/puncc/blob/main/docs/puncc_intro.ipynb


**Salesforce library** for anomaly detection & forecasting: https://github.com/salesforce/Merlion

<img width="503" alt="1" src="https://github.com/user-attachments/assets/e076da9d-425e-4a21-b547-abe10982e484">


**Multivariate Time-series**

For multivariate time-series data, one can follow VAR (vector autoregression) approach. 


One can utilize deep learning algorithms like LSTM (with either tensorflow or pytorch) in multivariate time-series.

LSTM with tensorflow: https://colab.research.google.com/drive/1B4OR2rHNjqOI6yKscouiik970_1OzeWf

**Probabilistic modeling**

For probabilistic time-series modeling, one can use **AutoGluon**: https://github.com/awslabs/gluonts

Chronos is a family of pretrained time series forecasting models. Chronos models are based on language model architectures, and work by quantizing time series into buckets which are treated as tokens: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html

From Amazon Science: https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting


Gluonts demo: https://colab.research.google.com/drive/1M7J9zSAJ6x6w-76JzOBjft3CvmCGaELP

Bayesian forecasting using statsmodels: https://github.com/ChadFulton/scipy2022-bayesian-time-series



**Foundation Model for timeseries forecasting**

Blog: https://blog.salesforceairesearch.com/moirai/

Repo: https://github.com/SalesforceAIResearch/uni2ts

TimeGPT by **NIXTLA**: https://github.com/Nixtla/nixtla

TimeGPT is production ready pre-trained time-series foundation model for forecasting & anomaly detection.





