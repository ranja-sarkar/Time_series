# timeseries

Time-series analysis is interesting, however forecasting with time-series data can be challenging. 
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
