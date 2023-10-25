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

### DEFINE CLASS FOR BLOCKING TIME SERIES CROSS VALIDATION ###
# create blocked time series class 
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits, tr_size=24, tt_size=6):
        self.n_splits = n_splits
        self.tr_size = tr_size
        self.tt_size = tt_size
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = self.tr_size + self.tt_size
        indices = np.arange(n_samples)

        margin = 0
        for i in np.linspace(0,n_samples-k_fold_size,self.n_splits, dtype='int'):
            start = i
            stop = start + k_fold_size
            mid = start + (k_fold_size - self.tt_size) 
            yield indices[start: mid], indices[mid + margin: stop]

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
def refit_sarimax(zc, endog_train, endog_test,exog_train = None, exog_test = None):
    '''
    Input: 
        fit : a previously fitted auto-arima model with initial starting params
        zc : target zipcode
        endog_train : training portion of the target time series variable
        endog_test : testing portion of target time series

        exog_train : (Optional) training portion of the feature covariate time series
        exog_test : (Optional) testing portino of the feature covariate time series
        
    Return:
        Best Parameters for SARIMAX Model along with cross validation scores
    '''
    ans = []

    # autoarima
    d = ndiffs(endog_train, alpha = 0.05)
    #D = nsdiffs(endog_train, m = 12)
    D = 0
    sxmodel = pm.auto_arima(y = endog_train, X = exog_train,
                            start_p=1, d = d, start_q=1, max_p=3, max_q=3, m=12,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, D = D,
                            seasonal=True, trace=True, error_action='ignore',
                            suppress_warnings=True, stepwise=True, information_criterion='bic')
    mape, mae, mse = mod_eval(sxmodel, endog_test, exog_test)
    comb = sxmodel.order
    combs = sxmodel.seasonal_order
    ans.append([zc, comb, combs, sxmodel.aic(), sxmodel.bic(),mape, mae, mse])
    
    # Convert into dataframe
    ans_df = pd.DataFrame(ans, columns=['zipcode', 'pdq', 'pdqs', 'aic', 'bic','MAPE', 'MAE', 'MSE'])
    
    return [ans_df, sxmodel]

def arima_findpdqs(y, X=None):
    # Pre-compute d and D
    d = ndiffs(y, alpha = 0.05)
    D = nsdiffs(y, m = 12)
    sxmodel = pm.auto_arima(y = y, X = X,
                            start_p=1, d = d, start_q=1, max_p=3, max_q=3, m=12,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, D = D,
                            seasonal=True, trace=True, error_action='ignore',
                            suppress_warnings=True, stepwise=True, information_criterion='bic')
    return([sxmodel.order, sxmodel.seasonal_order, sxmodel])

def autoarima_cv(zc, df, exog_var_names = None, type = 'Benchmark', traintest_cutoff = ['2022-12-01','2023-01-01'], splits=4, train_size=24, test_size=6):
    # Subset zipcode zc
    data = df[df['census_zcta5_geoid'] == zc].set_index('date')
    data = data.asfreq('MS')
    data_trainval = data.loc[:traintest_cutoff[0]]
    y_trainval = data.loc[:traintest_cutoff[0]]['sfr_rental_index']
    y_holdout = data.loc[traintest_cutoff[1]:]['sfr_rental_index']
    if exog_var_names:
        X_trainval, X_holdout = data.loc[:traintest_cutoff[0]][exog_var_names], data.loc[traintest_cutoff[1]:][exog_var_names]
    else:
        X_trainval, X_holdout = None, None
    # Get suggested order params from pm auto arima on training data and initial model
    order, seasonal_order, fitfull = arima_findpdqs(y = y_trainval, X=X_trainval)

    # cross_validate
    cv = BlockingTimeSeriesSplit(splits, tr_size=train_size, tt_size=test_size)

    res_sari_L = [] # instantiate empty list for storing

    for i , (tr, tt) in enumerate(cv.split(y_trainval)):
        y_train, y_test = y_trainval.iloc[tr], y_trainval.iloc[tt]
        
        if exog_var_names:
            X_train, X_test = X_trainval.iloc[tr], X_trainval.iloc[tt]
        else:
            X_train, X_test = None, None
            # refit model using new training and testing indices and autoarima in refit_sarimax function
        [res_sari, fit] = refit_sarimax(zc, endog_train = y_train, endog_test = y_test, 
                                  exog_train = X_train, exog_test = X_test)
        res_sari['crossfold'] = i
        res_sari['model'] = type
        res_sari_L.append(res_sari)
    # Get Final Holdout Test Score
    cols = pd.concat(res_sari_L).columns
    mape, mae, mse = mod_eval(fitfull, y_holdout, X_holdout, n_periods = 6)
    test_res = pd.DataFrame({cols[0]:zc, cols[1]:[fitfull.order], cols[2]:[fitfull.seasonal_order], cols[3]:fitfull.aic(), cols[4]:fitfull.bic(),cols[5]:mape, cols[6]:mae, cols[7]:mse, cols[8]:'test score', cols[9]: 'Holdout Test ' + type})
    res_sari_L.append(test_res)
    res_cv = pd.concat(res_sari_L)
    return(res_cv)


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
df = df.loc[~df['census_zcta5_geoid'].isin(zcs_to_rm)]
zcs = df.census_zcta5_geoid.unique()

################################ CROSS VALIDATION / FIT MODEL Univariate ##############################################
uni_ = []
for zc in zcs:
    univar_res = autoarima_cv(zc, df, exog_var_names = None)
    uni_.append(univar_res)
univar_df = pd.concat(uni_)
univar_df['mmape'] = univar_df.groupby(['zipcode', 'model'])['MAPE'].transform('mean')
plot_df = univar_df[['zipcode', 'model', 'mmape']].drop_duplicates()
univar_df.to_pickle('../data/sarimax_benchmark_cv_results.pkl')

sns.displot(plot_df, x="mmape", hue="model", kind="kde")

####################################### CV / FIT MULTIVARIATE + MFR_OCC###################################################
X_var_names = ['mfr_occ']
multi_ = []
for zc in zcs:
    multi_res = autoarima_cv(zc, df, exog_var_names=X_var_names, type = '+ MFR OCC')
    multi_.append(multi_res)

multi_df = pd.concat(multi_)
multi_df['mmape'] = multi_df.groupby(['zipcode', 'model'])['MAPE'].transform('mean')
multi_df.to_pickle('../data/sarimax_mfrocc_cv_results.pkl')



####################################### CV / FIT MULTIVARIATE + MFR_RENT###################################################
X_var_names = ['mfr_mean_rent']
multi_mfrrent = []
for zc in zcs:
    multi_res = autoarima_cv(zc, df, exog_var_names=X_var_names, type = '+ MFR Rent')
    multi_mfrrent.append(multi_res)

multi_mfrr_df = pd.concat(multi_mfrrent)
multi_mfrr_df.to_pickle('../data/sarimax_mfrrent_cv_results.pkl')

####################################### CV / FIT MULTIVARIATE + MFR_RENT###################################################
X_var_names = ['mfr_occ', 'mfr_mean_rent']
multi_ = []
for zc in zcs:
    multi_res = autoarima_cv(zc, df, exog_var_names=X_var_names, type = '+ MFR OCC & Rent')
    multi_.append(multi_res)

multi_mfrro_df = pd.concat(multi_)

multi_mfrro_df.to_pickle('../data/sarimax_mfrrentocc_cv_results.pkl')


####################################### Group and summarise #######################################
final = pd.concat([univar_df, multi_df, multi_mfrr_df, multi_mfrro_df])
final[['maic', 'mbic', 'mmape', 'mmae', 'mmse']] = final.groupby(['zipcode', 'model'])[['aic', 'bic', 'MAPE', 'MAE', 'MSE']].transform('mean')

final.drop(columns = ['aic', 'bic', 'MAPE', 'MAE', 'MSE', 'crossfold'], inplace = True)
final.drop_duplicates(inplace = True)
final.to_pickle('../data/pmdarima_cv_res_oct24.pkl')
######################################## PLOT #######################################

# create a seaborn plot
sns.set(style="darkgrid")

ax = sns.displot(final, x="mmape", hue="model", kind="kde")
# save the plot as JPG file
plt.savefig("../sarimax_cv_res.jpg", dpi=300)

