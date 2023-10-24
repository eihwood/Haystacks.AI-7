# General
import pickle
import pandas as pd
import numpy as np
import math
import datetime
from scipy import stats
import itertools
from pandas.plotting import register_matplotlib_converters
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer, make_column_selector

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Torch
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

# Set options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
warnings.filterwarnings('ignore')

# Load data, sort on zip and date and set index to datetime
with open("../data/sfr_mfr_mig_pre-processed.pkl", "rb") as f: df = pickle.load(f)
df.sort_values(['census_cbsa_geoid', 'census_zcta5_geoid', 'date'], inplace = True)

# Get relevant dates
df = df[(df['date']>'2014-12-01')& (df['date'] <= '2023-06-01')]

# drop incomplete zips fro MFR data
# drop incomplete zips
drops = df.loc[df.mfr_occ.isna()].census_zcta5_geoid.unique().tolist()
df.drop(df.loc[df['census_zcta5_geoid'].isin(drops)].index, inplace=True)

# Get relevant columns
df = df[['date', 'census_zcta5_geoid', 
         'sfr_rental_delta', 'sfr_price_delta', 
         'mfr_rental_delta', 'mfr_occ_delta',
        'sin_month', 'cos_month']]


# We want to know min / max values per offset columns (keep track for scaling -1, 1). When you get prediciton, scale it back
# Do min/max only on training dataset otherwise leaking info. 
sfr_delta_min = df['sfr_rental_delta'].min() 
sfr_delta_max = df['sfr_rental_delta'].max()
print('sfr delta min: ', sfr_delta_min)
print('sfr delta max: ', sfr_delta_max)

sfp_delta_min = df['sfr_price_delta'].min()
sfp_delta_max = df['sfr_price_delta'].max()
print('sfp delta min: ', sfp_delta_min)
print('spf delta max: ', sfp_delta_max)

mfr_delta_min = df['mfr_rental_delta'].min()
mfr_delta_max = df['mfr_rental_delta'].max()
print('mfr delta min: ', mfr_delta_min)
print('mfr delta max: ', mfr_delta_max)

mfo_delta_min = df['mfr_occ_delta'].min()
mfo_delta_max = df['mfr_occ_delta'].max()
print('mfo delta min: ', mfo_delta_min)
print('mfo delta max: ', mfo_delta_max)


# Define scaler
scaler = MinMaxScaler(feature_range=(-1, 1))

col_transform = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(-1, 1)), make_column_selector(dtype_include=np.number))], 
        remainder='passthrough',
        verbose_feature_names_out = False
)


###################### DEFINE CLASS ###################
# Define class
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
class SFR_DATASET(Dataset):
    def __init__(self, df, Ntrain, Npred, encoder):
        self.data = df.to_dict('records') # random access is easier with dictionaries
        self.encoder = encoder
        self.Ntrain = Ntrain
        self.Npred = Npred
    def __len__(self): 
        return len(self.data) - self.Ntrain - self.Npred  # subtract length of input + output
    
    def __getitem__(self, idx): 
        
        input = pd.DataFrame(self.data[idx:idx + self.Ntrain])
        output = pd.DataFrame(self.data[idx + self.Ntrain : idx + self.Ntrain + self.Npred])
        
        # each of these are 12x1 tensors (12 months of data)
        #in_zip = self.encoder.fit_transform([[zipcode] for zipcode in input['census_zcta5_geoid'].values])
        #in_zip = torch.tensor(in_zip, dtype=torch.float32)  #
        #in_zip = in_zip.view(-1) # This makes it so they can concatenate below
        in_zip = self.encoder.fit_transform([input['census_zcta5_geoid'].values])
        in_zip = torch.tensor(in_zip.flatten(), dtype=torch.float32)  # Flatten the 2D array
        in_sfr = torch.tensor(input['sfr_rental_delta']) 
        in_sfp = torch.tensor(input['sfr_price_delta'])
        in_mfr = torch.tensor(input['mfr_rental_delta'])
        in_mfo = torch.tensor(input['mfr_occ_delta'])
        in_sinm = torch.tensor(input['sin_month'])
        in_cosm = torch.tensor(input['cos_month']) 
        x = torch.cat((in_zip, in_sfr, in_sfp, in_mfr, in_mfo, in_sinm, in_cosm),dim = 0) # concat into 72-wide vector
        y = torch.tensor(output['sfr_rental_delta'])
        return {'X':x.float(), 'Y':y.float()}

def get_date_cutoff(dates, Ntrain, Npred):
    date_ = dates.unique()
    cutoffidx = len(date_) - Ntrain - Npred - 1
    cutoff_date = date_[cutoffidx]
    return cutoff_date


train_test_cutoff = get_date_cutoff(df.date, 12, 6)

# Cast the class, separating out by zipcdoe
# Create zip dict to store class for each zip
zip_train = []
zip_test = []

# Cast the class, separating out by zipcdoe
for zipcode in df['census_zcta5_geoid'].unique():
    
    # Filter for single zipcode
    zipcode_df = df[df['census_zcta5_geoid'] == zipcode]
    
    # Train test split
    df_train = zipcode_df.loc[(df.date < train_test_cutoff)]
    df_test = zipcode_df.loc[(df.date >= train_test_cutoff)]
    
    # Transform training data, cast class and store
    train_X = col_transform.fit_transform(df_train)
    train_X = pd.DataFrame(train_X, columns = col_transform.get_feature_names_out())
    train_sfr = SFR_DATASET(train_X, 12, 6, encoder)
    zip_train.append(train_sfr)
    
    # Transform testing data, cast class and store
    test_X = col_transform.fit_transform(df_test)
    test_X = pd.DataFrame(test_X, columns = col_transform.get_feature_names_out())
    test_sfr = SFR_DATASET(test_X, 12, 6, encoder)
    zip_test.append(test_sfr)


train_sfr = torch.utils.data.ConcatDataset(zip_train)
test_sfr = torch.utils.data.ConcatDataset(zip_test)
# check contents 
lens = [len(dataset) for dataset in train_sfr.datasets]
pd.Series(lens).unique() # 65 ... what does this 65 signify?
len(train_sfr.datasets) # is 181 (one dataset for each of the 181 zipcodes)


# Model - simple multilayer perceptron
# sequential 3 layer model
# want hidden dim between input and output for what we're doing or it can be less
# maximum number of free vars that actually alter anything. 
# can try diff hidden dims (bigger than or smaller than output dim)
# input -- hidden
# hidden -- output
# Model: simple multilayer perceptron
# sequential 3 layer model with 1 hidden dim

class SFR_MODEL(nn.Module):
    def __init__(self, indim, hdim, outdim):
        super().__init__() # for nn.MOdule you must initialize the super class
        self.layers = nn.Sequential(
            nn.Linear(indim, hdim),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(hdim, outdim)#,
            #nn.Tanh()
        )
    def forward(self, x):
        return self.layers(x)

# create instance of model
# indim matches length of input vector
# outdim matches length of output vector
# hdim?
model = SFR_MODEL(indim = 84, hdim = 36, outdim = 6)

model.train() # This is one place to set model model to "train' to introduce randomness
print(model)

# set up training loop
# optimizer - Adam good to start with
# datalaoder - wrap around dataset to shuffle through batches

opt = Adam(model.parameters()) # this is minimum (telling Adam all the numbers it can vary)
batchsize = 3
epochs = 150 # go though data 3x
loss_fn = nn.MSELoss()

# create dataloader
dl = DataLoader(train_sfr, batch_size = batchsize, shuffle = True, drop_last = True)
    
# initialize list for losses
losses_train = []
losses_test = []

# Loop through training data, train on it. Then loop through test data and then test on it. Do this within a single epoch. 
for epoch in trange(epochs):
    model.train()
    for batch in dl:
        opt.zero_grad() # at the beginning of batch, zero out the optimizer
        # use inputs and outputs to make model prediction
        x = batch['X']
        y = batch['Y']
        y_hat = model(x)
        
        loss = loss_fn(y_hat, y) # calculate loss
        loss.backward() # calculate gradient of loss
        opt.step() # runs the optimizer and updates model params based on gradient
        losses_train.append(loss.cpu().detach().numpy()) # single value as a numpy
    # Get y-test, apply dataloader, and set model to evaluate
    test_dl = DataLoader(test_sfr, batch_size = batchsize, shuffle = True)
    model.eval()
    for batch in test_dl:
        with torch.no_grad():
            x = batch['X']
            y = batch['Y']
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            losses_test.append(loss.cpu().detach().numpy())
                
            if losses_test[-1] == min(losses_test):
                torch.save(model, 'mlp_model.pt')
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss.cpu().detach().numpy(),
                           }, 'model_info.pt')


        
    




