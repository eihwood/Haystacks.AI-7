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

#Model Eval
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm  # Import tqdm for the progress bar


# Set options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


################################## DEFINE HELPER FUNCTIONS #################################################
# Define MAPE Function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Model Evaluation Function
def mod_eval(fit, endog_test, exog_test, start = '2023-01-01', end = '2023-06-01'):
    pred = fit.get_prediction(start = start, end = end, exog = exog_test)
    y_hat = pred.predicted_mean
    mape = mean_absolute_percentage_error(endog_test, y_hat)
    mae = mean_absolute_error(endog_test, y_hat)
    mse = mean_squared_error(endog_test, y_hat)
    res = [mape, mae, mse]
    return res

# Define function
def sarimax_gridsearch(zc, endog_train, exog_train, endog_test, exog_test, pdq, pdqs, maxiter=1000, freq='M'):
    '''
    Input: 
        endog_train : training portion of the target time series variable
        exog_train : training portion of the feature covariate time series
        endog_test : testing portion of target time series
        exog_test : testing portino of the feature covariate time series
        pdq : ARIMA order combinations for grid search
        pdqs : seasonal ARIMA order combinations for grid search
        maxiter : number of iterations, increase if your model isn't converging
        frequency : default='M' for month. Change to suit your time series frequency
            e.g. 'D' for day, 'H' for hour, 'Y' for year. 
        
    Return:
        Prints out top 5 parameter combinations
        Returns dataframe of parameter combinations ranked by BIC
    '''

    # Run a grid search with pdq and seasonal pdq parameters and get the best BIC value
    ans = []
    print(zc)
    pbar = tqdm(total=len(pdq) * len(pdqs))  # Initialize the progress bar
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = sm.tsa.statespace.SARIMAX(endog=endog_train, exog = exog_train,
                                                order=comb,
                                                seasonal_order=combs,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                fit = mod.fit(maxiter=maxiter, disp = False) 

                mape, mae, mse = mod_eval(fit, endog_test, exog_test)

                
                ans.append([zc, comb, combs, fit.bic, mape, mae, mse])
                print('SARIMAX {} x {}12 : BIC Calculated ={} : MAPE={}'.format(comb, combs,fit.bic, mape))
            except:
                continue
            finally:
                pbar.update(1)  # Update the progress bar

            
    # Find the parameters minimizing loss fn of choise
    pbar.close()  # Close the progress bar when done

    # Convert into dataframe
    ans_df = pd.DataFrame(ans, columns=['zipcode', 'pdq', 'pdqs', 'bic', 'MAPE', 'MAE', 'MSE'])

    # Sort and return top 5 combinations
    ans_df = ans_df.sort_values(by=['MAPE'],ascending=True)[0:5]
    
    return ans_df



######################## Define Parameter Ranges to Test ####################################################
# Note: higher numbers will result in code taking much longer to run
# Here we have it set to test p,d,q each = 0, 1, 2, 3, 4

# Define the p, d and q parameters to take any value between 0 and 3 (exclusive)
p = range(0,3)
q = range(0,4)
d = range(1,2)



# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
# 12 in the 's' position indicates monthly data

pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    


# Load data, sort on zip and date and set index to datetime
with open("../data/sfr_mfr_mig_pre-processed.pkl", "rb") as f: df = pickle.load(f)
df.sort_values(['census_cbsa_geoid', 'census_zcta5_geoid', 'date'], inplace = True)
# Subset relevant columns
df = df[['date', 'census_cbsa_geoid', 'census_zcta5_geoid', 'sfr_rental_index',
       'sfr_price_index', 'mfr_mean_rent_index', 'mfr_mean_occ_index']]

df = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2023-06-01')]

df['Month'] = df['date'].dt.month
# Define month transform function
df['sin_month'] = df['Month'].apply(lambda m: math.sin(2 * math.pi * ((m-1) / 11)))
df['cos_month'] = df['Month'].apply(lambda m: math.cos(2 * math.pi * ((m-1) / 11)))



# Get zipcodes that have full MFR data starting in 2015
missing_data = df[df['mfr_mean_rent_index'].isnull()]
zcs_to_rm = missing_data['census_zcta5_geoid'].unique().tolist()

# Filter rows where the 'zipcode' column is NOT in nan_zipcodes
df_filtered = df.loc[~df['census_zcta5_geoid'].isin(zcs_to_rm)]

zcs = df_filtered.census_zcta5_geoid.unique()

# Set holdout test 6 months for forecast
training_sta = '2015-01-01'
training_end = '2022-12-01'
test_sta = '2023-01-01'
test_end = '2023-06-01'

exog_var_names = ['sfr_price_index', 'mfr_mean_rent_index', 'mfr_mean_occ_index', 'sin_month', 'cos_month']
##############################################################################

# Iterate over zipcodes, running sarimax gridsearch
df_list = []
for zc in zcs:
    # Subset one zipcode
    data = df[df['census_zcta5_geoid'] == zc].set_index('date')
    data = data.asfreq('MS')
    # Train data
    endog_train = data.loc[training_sta:training_end, 'sfr_rental_index']
    exog_train = data.loc[training_sta:training_end, exog_var_names]
    # test data
    endog_test = data.loc[test_sta:test_end, 'sfr_rental_index']
    exog_test = data.loc[test_sta:test_end, exog_var_names]
    # Grid Search
    res_sari = sarimax_gridsearch(zc, endog_train, exog_train, endog_test, exog_test, pdq, pdqs, maxiter=200)
    df_list.append(res_sari) 


# Concatenate results together
final = pd.concat(df_list)
<<<<<<< HEAD

#final.to_csv('../data/sarimax_par_tuning_results1.csv')
final.to_pickle('../data/sarimax_par_tuning_results.pkl')




=======
>>>>>>> abf7715111350f966ee780b225fea4f324f8985d
