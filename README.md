
**Substack article on time-series analysis:**

https://open.substack.com/pub/ranjas/p/forecasting-with-time-series

<img width="598" alt="0" src="https://github.com/user-attachments/assets/adc097ef-5bef-4730-95ff-028a0bff3613">


Heteroscedasticity happens when the standard errors of a variable, monitored over a specific amount of time are non-constant. Conditional heteroscedasticity identifies non-constant volatility (degree of variation of series over time) when future periods of high & low volatility cannot be identified. Unconditional heteroscedasticity is used when future high & low volatility periods can be identified.

Some points to keep in mind while dealing with time-series data:

1) The uncertainty of forecast is just as important as the (point) forecast itself.
2) Model serving (deploying & scoring) is challenging.
3) Cross-validation (with sliding or expanding window as testing strategy) is tricky.
<img width="491" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/0dcf3f84-cc5c-4b16-98a5-58c4f2cddc9e">

In general, some time-series exhibit ill-behaved uncertainty. The forecast errors do not follow known distributions. Such information is useful for making judgmental decisions, but cannot be modeled and used for forecasting. Such an uncertainty is coconut uncertainty - of unknown unknowns leading to unpredictability. 

<img width="334" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/05b48529-29cc-4166-95ee-795a20961b5a">


Other time-series exhibit well-behaved uncertainty. The forecast errors follow known distributions - Normal, Poisson etc.. Such information is useful for modeling and predictions bound within a certain range. This window of uncertainty is subway uncertainty - of known unknowns. 

<img width="341" alt="image" src="https://github.com/ranja-sarkar/timeseries/assets/101544669/a6c5417f-50e5-4bd8-aefc-18bab5809f8a">

Forecast of level + trend is a baseline forecast. Baseline forecasts with the persistence model (using an observation at the previous time step to learn what will happen in the next time step)
quickly indicate whether you can do significantly better. 
The human mind is hardwired to look for patterns everywhere and we must be vigilant if we're developing elaborate models for random walk processes. A random walk having a step size that varies according to a normal distribution is used as a model for real-world time series data such as financial markets.

https://en.wikipedia.org/wiki/Random_walk

When time-series is treated as a signal, fast Fourier transform (FFT) is typically used to decompose the series and comprehend the presence of seasonality in it. 
For more, refer to: https://www.datacamp.com/datalab/w/274bcd35-70c7-42e7-9433-f967c5e3beb8#basic-components-of-a-fourier-term


**Approaches to smoothing a time-series: Baseline models**

**Holt's method** - there're level smoothing constant (alpha) and trend constant (beta).

**Holt Winter's method** - there's seasonal smoothing constant (delta) and considers seasonal baseline which is a regularly recurring pattern (day, week, month, quarter etc.) and baseline rises and falls at regular intervals. Deviation of each season from the baseline’s long-term (annual) average is used for forecasts. 

Smoothing models are for removal of noise. **Moving averages** are considered for these and they can be **simple**, **exponential**, and **cumulative**.
Example: https://www.kaggle.com/code/ranja7/energy-consumption-forecast-baseline-models


Simple Moving Average (SMA) uses a sliding window to take the average over a set number of time periods. It is an equally weighted mean of the previous data in this period.
Unlike SMA which drops the oldest observation as the new one gets added, cumulative moving average (CMA) considers all prior observations. 
Unlike SMA and CMA, exponential moving average (EMA) gives more weight to the recent prices as a result of which, EMA can better capture the movement of the trend. Exponential smoothing defines trend as the difference between observed values in consecutive records in time.

**Forecasting approach**

**ARIMA** handles data with trend.
**SARIMA** handles data with a seasonal component. 
The trend, seasonality and noise in a time series are explained by model parameter set (p,d,q), also called the order. The auto-regressive parameter is p; d is difference parameter and q is the moving average parameter. Trend is the long-term change in the mean level of observations. Seasonality is the pattern that’s periodically repeated, and noise is the random variation in the data. 

A time series is additive when the 'trend' is linear (changes in mean and variance are at linear rate) and 'seasonality' is constant in time. A time series is multiplicative when the 'trend' is non-linear. A stationary time series has constant mean over time and does not exhibit a trend. Having no trend means identically distributed random variables.  

Y(t) = Level + Trend + Seasonality + Noise 

Example: https://www.kaggle.com/code/ranja7/sarima-forecasts-auto-arima


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

skforecast: https://skforecast.org/0.13.0/index.html

tslearn: https://github.com/tslearn-team/tslearn

autoTS: https://github.com/winedarksea/AutoTS


Please note this 'autoTS' is different from 'auto_ts' used here:
https://colab.research.google.com/drive/1OgvVA1XLRle3gUBtu2tVcHmSTqR1-vP3?usp=sharing


Outliers in time-series data which are deviations from 'normal' called anomalies can be point or collective (subsequent). 

https://www.kaggle.com/code/ranja7/anomaly-detection-in-timeseries-isolation-forest

One can also use **PROPHET (by Meta)** for anomaly detection and forecasting:
https://facebook.github.io/prophet/docs/outliers.html

**Conformal prediction over time**

Conformal anomaly detection in timeseries (demo) example: https://colab.research.google.com/drive/1iP36nrvTge18kdNZPOXjn5nje8SMj-bk?usp=sharing#scrollTo=Z0oMmL7djROr%2F

Details about conformal anomaly detection: https://deel-ai.github.io/puncc/anomaly_detection.html

Conformal Prediction: https://github.com/deel-ai/puncc

Conformal Regression & Classification:  https://github.com/deel-ai/puncc/blob/main/docs/puncc_intro.ipynb


**Salesforce library** for anomaly detection & forecasting: https://github.com/salesforce/Merlion

<img width="503" alt="1" src="https://github.com/user-attachments/assets/e076da9d-425e-4a21-b547-abe10982e484">


**Multivariate Time-series**

For multivariate time-series data, one can follow VAR (vector autoregression) approach. 

Example: https://www.kaggle.com/code/ranja7/multivariate-timeseries-forecasting-var

One can utilize deep learning algorithms like LSTM (with either tensorflow or pytorch) in multivariate time-series.

Example: https://www.kaggle.com/code/ranja7/forecasting-with-lstm-tensorflow


**Probabilistic modeling**

For probabilistic time-series modeling, one can use **AutoGluon**: https://github.com/awslabs/gluonts

Chronos is a family of pretrained time series forecasting models. Chronos models are based on language model architectures, and work by quantizing time series into buckets which are treated as tokens: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html

From Amazon Science: https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting


Demo: https://colab.research.google.com/drive/1M7J9zSAJ6x6w-76JzOBjft3CvmCGaELP?usp=sharing

Bayesian forecasting using statsmodels: https://github.com/ChadFulton/scipy2022-bayesian-time-series



**Foundation Model for timeseries forecasting**

Blog: https://blog.salesforceairesearch.com/moirai/

Repo: https://github.com/SalesforceAIResearch/uni2ts

TimeGPT by **NIXTLA**: https://github.com/Nixtla/nixtla

TimeGPT is production ready pre-trained time-series foundation model for forecasting & anomaly detection.





