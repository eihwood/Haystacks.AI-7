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
from pandas.plotting import register_matplotlib_converters
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

import torch.nn as nn


# torch
import torch

from torch.optim import Adam

# Dataset from torch
from torch.utils.data import Dataset, DataLoader

# Load data, sort on zip and date and set index to datetime
with open("../data/sfr_mfr_mig_pre-processed.pkl", "rb") as f: df = pickle.load(f)
df.sort_values(['census_cbsa_geoid', 'census_zcta5_geoid', 'date'], inplace = True)

df = df[['date', 'census_zcta5_geoid', 'sfr_rental_index', 'sfr_price_index', 'mfr_mean_rent_index', 'mfr_mean_occ_index']]
df = df[df['date']>='2014-12-01']

# Month as input
df['Month'] = df['date'].dt.month # make this cyclical so 11-12 is same as 12-1 (1-12) sin and cosin of the month 0-2pi


data = df[df['census_zcta5_geoid'] == '30002']

# Translate all index columns into delta
data['offset'] = data['sfr_rental_index'].shift(periods = 1)
data.dropna(inplace=True)
data['sfr_rental_delta'] = data['sfr_rental_index'] - data['offset'] 

# We want to know min / max values per offset columns (keep track for scaling -1, 1). When you get prediciton, scale it back
# Do min/max only on training dataset otherwise leaking info. 
sfr_offset_min = data['sfr_rental_delta'].min() # store as meta data for each indexed column
sfr_offset_max = data['sfr_rental_delta'].max() # store as meta data for each indexed column

# Define month transform function
data['sin_month'] = data['Month'].apply(lambda m: math.sin(2 * math.pi * ((m-1) / 11)))
data['cos_month'] = data['Month'].apply(lambda m: math.cos(2 * math.pi * ((m-1) / 11)))



# give it actual data, make it so that we can load random records, format as however we want for model (assume data is already normalized in the -1,1 range
# note: this is assuming just one zipcode. when we have many, we'll need to group by zipcode
class SFR_DATASET(Dataset):
    def __init__(self, df):
        self.data = df.to_dict('records') # random access is easier with dictionaries
    def __len__(self): 
        return len(self.data)-10 # how many records - every example is 9 months (every zipcode every month)
    def __getitem__(self, idx): # given an index, get three months, and then next six months out of total data (forward by one month)
        input = pd.DataFrame(self.data[idx:idx+3]) # assuming 3 columns (sfr and sin and cos of month)
        output = pd.DataFrame(self.data[idx+4:idx+10])
        in_sfr = torch.tensor(input['sfr_rental_delta'])
        in_sinm = torch.tensor(input['sin_month'])
        in_cosm = torch.tensor(input['cos_month']) # each of these are 3 months tensor (one dimensional vectors)
        x = torch.cat((in_sfr, in_sinm, in_cosm), dim = 0) # concateting into one big 9 wide vector
        y = torch.tensor(output['sfr_rental_delta'])
        return {'X':x.float(), 'Y':y.float()}

# try the casting the class
sfr = SFR_DATASET(data)


# Model - simple multilayer perceptron
# sequential 3 layer model
# want hidden dim between input and output for what we're doing or it can be less
# maximum number of free vars that actually alter anything. 
# can try diff hidden dims (bigger than or smaller than output dim)
# input -- hidden
# hidden -- output

class SFR_MODEL(nn.Module):
    def __init__(self, indim, hdim, outdim):
        super().__init__() # for nn.MOdule you must initialize the super class
        self.layers = nn.Sequential(
            nn.Linear(indim, hdim),
            nn.LeakyReLU(),
            nn.Linear(hdim, outdim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.layers(x)


# at this point, create instance of model with input 
model = SFR_MODEL(indim = 9, hdim = 4, outdim = 6)
print(model)


# the way to set up training loop... we need an optimizer and a data loader
# Adam good optimizer to start with
opt = Adam(model.parameters()) # this is minimum (telling Adam all the numbers it can vary)
batchsize = 3
epochs = 50 # through data once (but typically go through multiple times
# Dataloader is the thing we wrap around dataset to get a batchsize and do shuffling
dl = DataLoader(sfr, batch_size = batchsize, shuffle = True, drop_last = True) # take batchsize examples and group
loss_fn = nn.MSELoss()

# Training loop

losses = [] # initialize empty losses list to store loss values for plotting later

for epoch in range(epochs):
    print(epoch)
    for batch in dl:
        opt.zero_grad() # at teh beginning of batch, zero out the optimizer
        x = batch['X']
        y = batch['Y']
        y_hat = model(x)
        # need a loss function
        loss = loss_fn(y_hat, y) # computes the mean squared error loss between pred and target
        loss.backward() # calculates the gradient 
        opt.step() # runs the optimizer and updates model params based on gradient
        losses.append(loss.cpu().detach().numpy()) # single value as a numpy



# plot - if this was learning something, you'd want to see decrease
plt.plot(np.array(losses)) # it is generally trending down

# Next steps: things we should be able to handle on our own
# rest of input columns
# train test split 
# take a stab at integrating all zipcodes (e.g. make dict of zipcodes and then handle zipcodes one by one)
# not necessarily anything specific to what we are trying to do that needs either a cnn or rnn




        
    



