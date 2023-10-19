# General
import pickle
import pandas as pd
import numpy as np
import math
import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import itertools
from tqdm import tqdm  # Import tqdm for the progress bar
from sklearn.model_selection import TimeSeriesSplit

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")

# SARIMAX and Bayesian Estimation 
import statsmodels.api as sm #sarimax
import pmdarima as pm
from pmdarima.arima import ARIMA
from pmdarima.arima.utils import ndiffs
from pmdarima.arima.utils import nsdiffs

#Model Eval
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm  # Import tqdm for the progress bar


# Set options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


################################## DEFINE HELPER FUNCTIONS #################################################
# Model Evaluation Function
def mod_eval(sxmodel, endog_test, exog_test = None, n_periods = 6):
    y_hat = sxmodel.predict(n_periods = n_periods, X = exog_test, return_conf_int = False)

    mape = mean_absolute_percentage_error(endog_test, y_hat)
    mae = mean_absolute_error(endog_test, y_hat)
    mse = mean_squared_error(endog_test, y_hat)
    res = [mape, mae, mse]
    return res

# Define function
def fit_sarimax(zc, endog_train, endog_test, order, seasonal_order = (0,0,0,12),exog_train = None, exog_test = None):
    '''
    Input: 
        zc : target zipcode
        endog_train : training portion of the target time series variable
        exog_train : (Optional) training portion of the feature covariate time series
        endog_test : testing portion of target time series
        exog_test : (Optional) testing portino of the feature covariate time series
        
    Return:
        Best Parameters for SARIMAX Model along with cross validation scores
    '''

    # Run a grid search with pdq and seasonal pdq parameters and get the best BIC value
    ans = []
    print(zc)
    #pbar = tqdm(total=len(pdq) * len(pdqs))  # Initialize the progress bar

    
    # SARIMAX Modelt 
    sxmodel = ARIMA(order, seasonal_order, start_params=None, method='lbfgs', with_intercept = True, trend = 'c')
    sx_fit = sxmodel.fit(y = endog_train, X = exog_train)
    mape, mae, mse = mod_eval(sx_fit, endog_test, exog_test)
    comb = sxmodel.order
    combs = sxmodel.seasonal_order
    ans.append([zc, comb, combs, sxmodel.aic(), sxmodel.bic(),mape, mae, mse])
    
    # Convert into dataframe
    ans_df = pd.DataFrame(ans, columns=['zipcode', 'pdq', 'pdqs', 'aic', 'bic','MAPE', 'MAE', 'MSE'])
    
    return ans_df

def find_pdqs(y, X=None):
    # Pre-compute d and D
    d = ndiffs(y, alpha = 0.05)
    D = nsdiffs(y, m = 12)
    sxmodel = pm.auto_arima(y = y, X = X,
                            start_p=1, d = d, start_q=1, max_p=3, max_q=3, m=12,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, D = D,
                            seasonal=True, trace=True, error_action='ignore',
                            suppress_warnings=True, stepwise=True, information_criterion='bic')
    return([sxmodel.order, sxmodel.seasonal_order])

def allzips_cv(zcs, df_trainval, exog_var_names = None):
    df_list = []
    for zc in zcs:
        # Subset one zipcode
        data = df_trainval[df_trainval['census_zcta5_geoid'] == zc].set_index('date')
        data = data.asfreq('MS')
        y = data['sfr_rental_index']
        if exog_var_names:
            X = data[exog_var_names]
        # Get suggested order params from pm auto arima
        order, seasonal_order = find_pdqs(y = y)
        # Train, Test, Split
        tscv = TimeSeriesSplit(n_splits = 10, max_train_size=None, test_size = 6) # hold out 6 months as test set
        res_sari_L = []
        for i, (train_index, test_index) in enumerate(tscv.split(data)):
            print('TRAIN:', train_index, 'TEST:', test_index) 
            if exog_var_names:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train = None
                X_test = None
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Fit
            res_sari = fit_sarimax(zc, endog_train = y_train, endog_test = y_test, 
                                   order = order, seasonal_order = seasonal_order,
                                  exog_train = X_train, exog_test = X_test)
            res_sari['crossfold'] = i
            res_sari['type'] = 'Univariate SARIMA'
            res_sari_L.append(res_sari)
        res_cv = pd.concat(res_sari_L)
    df_list.append(res_cv) 
    return(pd.concat(df_list))


# Load data, sort on zip and date and set index to datetime
with open("../data/sfr_mfr_mig_pre-processed.pkl", "rb") as f: df = pickle.load(f)
df.sort_values(['census_cbsa_geoid', 'census_zcta5_geoid', 'date'], inplace = True)
# Subset relevant columns
df = df[['date', 'census_cbsa_geoid', 'census_zcta5_geoid', 'sfr_rental_index',
       'sfr_price_index', 'coef', 'nounits', 'occupied_units', 'mfr_occ',
       'mfr_mean_occ_index', 'mfr_mean_rent', 'mfr_mean_rent_index', 'month',
       'cos_month', 'sin_month', 'sfr_rental_delta']]

# Subset to where we have full MFR data
df = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2023-06-01')]


# Get zipcodes that have full MFR data starting in 2015
missing_data = df[df['mfr_mean_rent_index'].isnull()]
zcs_to_rm = missing_data['census_zcta5_geoid'].unique().tolist()

# Filter rows where the 'zipcode' column is NOT in nan_zipcodes
df_filtered = df.loc[~df['census_zcta5_geoid'].isin(zcs_to_rm)]
df_trainval = df_filtered[df_filtered['date']<= '2022-12-01']
zcs = df_filtered.census_zcta5_geoid.unique()

################################ CROSS VALIDATION / FIT MODEL Univariate ##############################################

univar_res = allzips_cv(zcs, df_trainval)

univar_res.to_pickle('../data/sarimax_cv_results.pkl')


####################################### MULTIVARIATE ###################################################
exog_var_names = ['sfr_price_index', 'mfr_mean_rent_index', 'mfr_mean_occ_index', 'sin_month', 'cos_month']



###################################### HOLDOUT TEST