# General
import pickle
import pandas as pd
import numpy as np
import datetime
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# SARIMAX
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Set options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


# Load data, sort on zip and date and set index to datetime
with open("../data/sfr_mfr_mig_pre-processed.pkl", "rb") as f: df = pickle.load(f)
df.sort_values(['census_cbsa_geoid', 'census_zcta5_geoid', 'date'], inplace = True)

# do some differencing on log data
df['ln_rpi'] = np.log(df['sfr_rental_index'])
df['D.ln_rpi'] = df['ln_rpi'].diff()

# Subset one zipcode
data = df[df['census_zcta5_geoid'] == '44333'].set_index('date')
data = data.asfreq('MS')

# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

# Levels
axes[0].plot(data.index._mpl_repr(), data['sfr_rental_index'], '-')
axes[0].set(title='Single Family Rent Price Index')

# Log difference
axes[1].plot(data.index._mpl_repr(), data['D.ln_rpi'], '-')
axes[1].hlines(0, data.index[0], data.index[-1], 'r')
axes[1].set(title='US Wholesale Price Index - difference of logs');


#From the first two graphs, we note that the original time series does not appear to be stationary, whereas the first-difference does. #This supports either estimating an ARMA model on the first-difference of the data, or estimating an ARIMA model with 1 order of #integration

# ACF AND PACF on stationary transformed data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(data.iloc[1:]['D.ln_rpi'], lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(data.iloc[1:]['D.ln_rpi'], lags=40, ax=axes[1])


# From the above, we chose AR parameter 'p' based on significant spikes in the PACF plot, in this case 1
# From the above, we chose MA parameter 'q' based on sharp cutoffs in the ACF plot at q lags. In this case we can 
# try q = 1
# Final ARIMA model: ARIMA(1,1,3)
ar = 1
ma = 3
mod = sm.tsa.statespace.SARIMAX(data['sfr_rental_index'], trend='c', order=(ar,1,ma))
res_fit = mod.fit(disp=False)
print(res_fit.summary())


# Plot results
predict_mle = res_fit.get_prediction()
predict_mle_ci = predict_mle.conf_int()
lower = predict_mle_ci["lower sfr_rental_index"]
upper = predict_mle_ci["upper sfr_rental_index"]

# Graph
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

# Plot data points
data['sfr_rental_index'].plot(ax=ax, style="-", label="Observed")

# Plot predictions
predict_mle.predicted_mean.plot(ax=ax, style="r.", label="One-step-ahead forecast")
ax.fill_between(predict_mle_ci.index, lower, upper, color="r", alpha=0.1)
ax.legend(loc="lower left")
ax.set_ylim(50,190)
plt.show()

mse = mean_squared_error(data['sfr_rental_index'],predict_mle.predicted_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(data['sfr_rental_index'],predict_mle.predicted_mean)
print('ARIMAX model Training MSE:{}'.format(mse))
print('ARIMAX model Training MAE:{}'.format(mae))
print('ARIMAX model Training RMSE:{}'.format(rmse))


#################################### Exogenous variables ####################################
# variables
training_sta = '2015-01-01'
training_end = '2022-06-01'
endog = data.loc[training_sta:training_end, 'sfr_rental_index']
exog = data.loc[training_sta:training_end, ['mfr_mean_rent_index', 'mfr_mean_occ_index']]

# Fit the model
mod = sm.tsa.statespace.SARIMAX(endog, exog, order=(ar,1,ma))
res = mod.fit(disp=False, maxiter = 1000)
print(res.summary())

# get predictions
predict_mle = res.get_prediction()
predict_mle_ci = predict_mle.conf_int()
lower = predict_mle_ci["lower sfr_rental_index"]
upper = predict_mle_ci["upper sfr_rental_index"]

# Graph
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

# Plot data points
data.loc[training_sta:training_end, 'sfr_rental_index'].plot(ax=ax, style="-", label="Observed")

# Plot predictions
predict_mle.predicted_mean.plot(ax=ax, style="r.", label="One-step-ahead forecast")
ax.fill_between(predict_mle_ci.index, lower, upper, color="r", alpha=0.1)
ax.legend(loc="lower left")
ax.set_ylim(50,190)
plt.show()

mse = mean_squared_error(data.loc[training_sta:training_end, 'sfr_rental_index'],predict_mle.predicted_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(data.loc[training_sta:training_end, 'sfr_rental_index'],predict_mle.predicted_mean)
print('ARIMAX model Training MSE:{}'.format(mse))
print('ARIMAX model Training MAE:{}'.format(mae))
print('ARIMAX model Training RMSE:{}'.format(rmse))


# Get test predictions
# get predictions
exog_test = data.loc['2022-07-01':'2023-06-01', ['mfr_mean_rent_index', 'mfr_mean_occ_index']]
endog_test = data.loc['2022-07-01':'2023-06-01', 'sfr_rental_index']
predict_mle = res.get_prediction(start = '2022-07-01', end = '2023-06-01', exog = exog_test)
predict_mle_ci = predict_mle.conf_int()
lower = predict_mle_ci["lower sfr_rental_index"]
upper = predict_mle_ci["upper sfr_rental_index"]
mse = mean_squared_error(data.loc['2022-07-01':'2023-06-01', 'sfr_rental_index'],predict_mle.predicted_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(data.loc['2022-07-01':'2023-06-01', 'sfr_rental_index'],predict_mle.predicted_mean)
print('ARIMAX model Test MSE:{}'.format(mse))
print('ARIMAX model Test MAE:{}'.format(mae))
print('ARIMAX model Test RMSE:{}'.format(rmse))

# BAG TESTING FOR HOLD OUT

# Graph
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

# Plot data points
data.loc['2022-07-01':'2023-06-01', 'sfr_rental_index'].plot(ax=ax, style="-", label="Observed")

# Plot predictions
predict_mle.predicted_mean.plot(ax=ax, style="r.", label="One-step-ahead forecast")
ax.fill_between(predict_mle_ci.index, lower, upper, color="r", alpha=0.1)
ax.legend(loc="lower left")
#ax.set_ylim(50,190)
plt.show()

# USE K-schiller as a benchmark - compare to market (zillow index e.g)


# ARIMA Postestimation Dynamic Forecasting
# Variables
endog = data.loc['2015-01-01':'2023-06-01', 'sfr_rental_index']
exog = data.loc['2015-01-01':'2023-06-01', ['mfr_mean_rent_index', 'mfr_mean_occ_index']]
nobs = endog.shape[0]

# Fit the model
mod = sm.tsa.statespace.SARIMAX(endog.loc[:'2022-06-01'], exog[:'2022-06-01'], order=(ar,1,ma))
fit_res = mod.fit(disp=False, maxiter = 1000)
print(fit_res.summary())

# Now get results for full dataset using estimated parameters (on a subset of the data)
mod = sm.tsa.statespace.SARIMAX(endog, exog = exog, order=(ar,1,ma))
res = mod.filter(fit_res.params)

# Get in sample predictions using predict command, full_results = True for CIs
# Returns the one-step-ahead-in-sample predictions for entire sample

# In-sample one-step-ahead predictions (uses the true value of the endogenous values at each step to predict the next in-sample value
predict = res.get_prediction()
predict_ci = predict.conf_int()

# DYNAMIC Predictions: use one-step-ahead up to some specified point in the time series, then use the previous predicted endogenous values in place of the true endogenous value
# Dynamic predictions
predict_dy = res.get_prediction(dynamic='2022-06-01')
predict_dy_ci = predict_dy.conf_int()

# graph the one-step-ahead and dynamic predictions (and the corresponding confidence intervals) to see their relative performance. Notice that up to the point where dynamic prediction begins (1978:Q1), the two are the same.

# Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='SFR Rental', xlabel='Date', ylabel='Rent Price Index')
ax.set_ylim(bottom = 100, top = 190)
# Plot data points
data.loc['2015-01-01':'2023-06-01', 'sfr_rental_index'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.loc['2015-01-01':'2023-06-01'].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['2015-01-01':'2023-06-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.loc['2015-01-01':'2023-06-01'].plot(ax=ax, style='g', label='Dynamic forecast (June 2022-Present)')
ci = predict_dy_ci.loc['2015-01-01':'2023-06-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)
legend = ax.legend(loc='lower right')

# ZOOM IN GRAPH

fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='SFR Rental', xlabel='Date', ylabel='Rent Price Index')
ax.set_ylim(bottom = 100, top = 190)

# Plot data points
data.loc['2021-06-01':'2023-06-01', 'sfr_rental_index'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.loc['2021-06-01':'2023-06-01'].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['2021-06-01':'2023-06-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.loc['2021-06-01':'2023-06-01'].plot(ax=ax, style='g', label='Dynamic forecast (June 2022-Present)')
ci = predict_dy_ci.loc['2021-06-01':'2023-06-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)
legend = ax.legend(loc='lower right')


# Graph the prediction error. It is obvious that, as one would suspect, one-step-ahead prediction is considerably better.

# Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

# In-sample one-step-ahead predictions and 95% confidence intervals
predict_error = predict.predicted_mean - endog
predict_error.loc['2021-06-01':'2023-06-01'].plot(ax=ax, label='One-step-ahead forecast')
ci = predict_ci.loc['2021-06-01':'2023-06-01'].copy()
ci.iloc[:,0] -= endog.loc['2021-06-01':'2023-06-01']
ci.iloc[:,1] -= endog.loc['2021-06-01':'2023-06-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.1)

# Dynamic predictions and 95% confidence intervals
predict_dy_error = predict_dy.predicted_mean - endog
predict_dy_error.loc['2021-06-01':'2023-06-01'].plot(ax=ax, style='r', label='Dynamic forecast (1978)')
ci = predict_dy_ci.loc['2021-06-01':'2023-06-01'].copy()
ci.iloc[:,0] -= endog.loc['2021-06-01':'2023-06-01']
ci.iloc[:,1] -= endog.loc['2021-06-01':'2023-06-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

legend = ax.legend(loc='lower left');
legend.get_frame().set_facecolor('w')

# In-sample one-step-ahead predictions, and out-of-sample forecasts
# Now get results for full dataset using estimated parameters (on a subset of the data)
# Fit the model
mod = sm.tsa.statespace.SARIMAX(endog.loc[:'2022-06-01'], exog.loc[:'2022-06-01'], order=(ar,1,ma))

res = mod.fit(disp=False, maxiter = 1000)

# In-sample one-step-ahead predictions, and out-of-sample forecasts
predict = res.get_prediction(start = '2015-01-01',end='2023-06-01', exog = exog_test)
idx = np.arange(len(predict.predicted_mean))
predict_ci = predict.conf_int()
ci = predict_ci.loc['2015-1-01':'2023-06-01'].copy()
ci.iloc[:,0] -= endog.loc['2015-1-01':'2023-06-01']
ci.iloc[:,1] -= endog.loc['2015-1-01':'2023-06-01']

# Graph
fig, ax = plt.subplots(figsize=(12,6))
ax.xaxis.grid()
ax.plot(endog, 'k.')
ax.ylim(bottom = 100, top = 200)
# Plot
ax.plot(predict.predicted_mean.loc['2015-01-01':'2022-06-01'], 'gray')
ax.plot(predict.predicted_mean.loc['2022-07-01':'2023-06-01'], 'k--', linestyle='--', linewidth=2)
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.15)




#################################### MODEL SELECTION ###########################################

