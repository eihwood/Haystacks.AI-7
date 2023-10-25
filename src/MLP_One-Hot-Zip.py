# General
import pickle
import pandas as pd
import numpy as np
import warnings
import json 

# Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector

# Torch
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from torchmetrics.regression import MeanAbsolutePercentageError
import torch.nn.functional as F

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
warnings.filterwarnings('ignore')

# THINGS TO TRY
# Change cutoff date so we have more than one sample in the test dataset (e.g.) walk it back 9 months
# Try a smaller hdim (2, 16)
# 3 months predict instead of 6 months prediction


############################################# LOAD DATA ###############################################
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

# Create zip dict with keys and tensors for one-hot encoding
keys = df['census_zcta5_geoid'].to_list()
zip_dict = {k: i for i, k in enumerate(set(keys))}

# save out as json
with open('../zipcode_onehot_dict.json', 'w') as fp:
    json.dump(zip_dict, fp)


########################################## DEFINE SCALER, CLASS ######################################

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

class SFR_DATASET(Dataset):
    def __init__(self, df, Ntrain, Npred, zip_dict, zc):
        self.data = df.to_dict('records') # random access is easier with dictionaries
        self.zip_dict = zip_dict
        self.Ntrain = Ntrain
        self.Npred = Npred
        self.zc = zc
    def __len__(self): 
        return len(self.data) - self.Ntrain - self.Npred  # subtract length of input + output
    
    def __getitem__(self, idx): 

        input = pd.DataFrame(self.data[idx:idx + self.Ntrain])
        output = pd.DataFrame(self.data[idx + self.Ntrain : idx + self.Ntrain + self.Npred])

        # Get zip dict tensor as tensor
        # each of these are 12x1 tensors (12 months of data)
        in_zip = F.one_hot(torch.tensor(self.zip_dict[self.zc]), len(self.zip_dict.keys()))
        in_sfr = torch.tensor(input['sfr_rental_delta']) 
        in_sfp = torch.tensor(input['sfr_price_delta'])
        in_mfr = torch.tensor(input['mfr_rental_delta'])
        in_mfo = torch.tensor(input['mfr_occ_delta'])
        in_sinm = torch.tensor(input['sin_month'])
        in_cosm = torch.tensor(input['cos_month']) 
        x = torch.cat((in_zip, in_sfr, in_sfp, in_mfr, in_mfo, in_sinm, in_cosm),dim = 0) # concat into 72-wide vector
        y = torch.tensor(output['sfr_rental_delta'])
        return {'X':x.float(), 'Y':y.float(), 'zipcode':self.zc}

def get_date_cutoff(dates, Ntrain, Npred):
    date_ = dates.unique()
    cutoffidx = len(date_) - Ntrain - Npred - 10 # if we want 9 extra months in test
    cutoff_date = date_[cutoffidx]
    return cutoff_date

train_test_cutoff = get_date_cutoff(df.date, 12, 6)

# Cast the class, separating out by zipcdoe
# Transform data
# Train test split
df_train = df.loc[(df.date < train_test_cutoff)]
df_test = df.loc[(df.date >= train_test_cutoff)]
train_X = col_transform.fit_transform(df_train)
train_X = pd.DataFrame(train_X, columns = col_transform.get_feature_names_out())

# Create dict of min and max delta values to be able to back transform model results

# Do min/max only on training dataset otherwise leaking info. 
scaling_dict = {'sfrMin': df_train['sfr_rental_delta'].min(),
               'sfrMax': df_train['sfr_rental_delta'].max(),
               'sfpMin': df_train['sfr_price_delta'].min(),
               'sfpMax': df_train['sfr_price_delta'].max(),
               'mfrMin': df_train['mfr_rental_delta'].min(),
               'mfrMax': df_train['mfr_rental_delta'].max(),
               'mfoMin': df_train['mfr_occ_delta'].min(),
               'mfoMax': df_train['mfr_occ_delta'].max()}


# save out as json
with open('../scaling_dict.json', 'w') as fp:
    json.dump(scaling_dict, fp)

# Transform test data separately
test_X = col_transform.fit_transform(df_test)
test_X = pd.DataFrame(test_X, columns = col_transform.get_feature_names_out())

# Cast the class, separating out by zipcdoe
# Create zip list to store class for each zip
zip_train = []
zip_test = []

for zipcode in df['census_zcta5_geoid'].unique():
    
    # Filter for single zipcode
    train_zip_df = train_X[train_X['census_zcta5_geoid'] == zipcode]
    test_zip_df = test_X[test_X['census_zcta5_geoid'] == zipcode]
    
    # Transform training data, cast class and store
    train_sfr = SFR_DATASET(train_zip_df, 12, 6, zip_dict, zipcode)
    zip_train.append(train_sfr)
    
    # Transform testing data, cast class and store
    test_sfr = SFR_DATASET(test_zip_df, 12, 6, zip_dict, zipcode)
    zip_test.append(test_sfr)

train_sfr = torch.utils.data.ConcatDataset(zip_train)
test_sfr = torch.utils.data.ConcatDataset(zip_test)

# check contents 
print(len(test_sfr.datasets)) # is 181 (one dataset for each of the 181 zipcodes)
print(len(train_sfr.datasets[0])) # 56
print(len(test_sfr.datasets[0]))  # 10

# MAKE CUTOFF DATE EARLIER SO WE HAVE MORE OF A 80/20 train/test split

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
# Because of overfitting, cut hdim way down (e.g. 16 hdim) h is essentially degrees of freedom. Could even try 2 to see how it works (maybe)
model = SFR_MODEL(indim = 253, hdim = 16, outdim = 6)

model.train() # This is one place to set model model to "train' to introduce randomness
print(model)

# set up training loop
# optimizer - Adam good to start with
# datalaoder - wrap around dataset to shuffle through batches

opt = Adam(model.parameters()) # this is minimum (telling Adam all the numbers it can vary)
batchsize = 3
epochs = 30 # ideally want to train while test loss is still going down. if after a while, test levels off. 
loss_fn = nn.MSELoss()

# create dataloader for training set
train_dl = DataLoader(train_sfr, batch_size = batchsize, shuffle = True, drop_last = True)
# Apply dataloader to test set
test_dl = DataLoader(test_sfr, batch_size = batchsize, shuffle = True, drop_last=True)
    
# initialize list to store dictionaries of metrics
eval_train = []
eval_test = []
losses_test = [] # to compare losses to losses

# to store y and y_hat
preds_train = {}
preds_test = {}

# Loop through training data, train on it. Then loop through test data and then test on it. Do this within a single epoch. 
for epoch in trange(epochs):
    
    #TRAIN
    model.train()
    for i, batch in enumerate(train_dl):
        opt.zero_grad() # at the beginning of batch, zero out the optimizer
        # use inputs and outputs to make model prediction
        x = batch['X']
        y = batch['Y']
        y_hat = model(x)
        
        mean_abs_percentage_error = MeanAbsolutePercentageError()
        mape = mean_abs_percentage_error(y_hat, y)
        
        loss = loss_fn(y_hat, y) # calculate loss
        loss.backward() # calculate gradient of loss
        opt.step() # runs the optimizer and updates model params based on gradient
        
        for j, zcode in enumerate(batch['zipcode']):
            preds_train[zcode] = {}
            preds_train[zcode]['month'] = np.arange(1,7) # predict 6 months
            preds_train[zcode]['y'] = y[j].numpy()
            preds_train[zcode]['y_hat'] = y_hat[j].detach().numpy()
            preds_train[zcode]['epoch'] = epoch
            preds_train[zcode]['batch'] = j
            
        eval_train.append({'epoch': epoch, 'batch_num': i, 
                           'mape': mape.cpu().detach().numpy(),
                           'loss_mse': loss.cpu().detach().numpy()})
        
    # TEST    
    model.eval()
    for i, batch in enumerate(test_dl):
        with torch.no_grad():
            x = batch['X']
            y = batch['Y']
            zipcode = batch['zipcode']
            y_hat = model(x)
            
            mean_abs_percentage_error = MeanAbsolutePercentageError()
            mape = mean_abs_percentage_error(y_hat, y)
            
            loss = loss_fn(y_hat, y)
            losses_test.append(loss.cpu().detach().numpy())

            for j, zcode in enumerate(batch['zipcode']):
                preds_test[zcode] = {}
                preds_test[zcode]['month'] = np.arange(1,7) # predict 6 months
                preds_test[zcode]['y'] = y[j].numpy()
                preds_test[zcode]['y_hat'] = y_hat[j].detach().numpy()
                preds_test[zcode]['epoch'] = epoch
                preds_test[zcode]['batch'] = j
                
            eval_test.append({'epoch': epoch, 'batch_num': i, 
                              'mape': mape.cpu().detach().numpy(),
                              'loss_mse': loss.cpu().detach().numpy()})
            
            # could save each one on an epoch basis associated with a loss, partition out later... but this is fine for now   
            if losses_test[-1] <= min(losses_test):
                torch.save(model, 'mlp_model.pt')
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss.cpu().detach().numpy(),
                           }, 'model_info.pt')

# create dataframe of training/test loss and error
res_train, res_test = pd.DataFrame.from_dict(eval_train), pd.DataFrame.from_dict(eval_test)
res_train['Type'] = 'Train'
res_test['Type'] = 'Test'

res = pd.concat([res_train, res_test])
res['Model'] = 'MLP-ziponehot'
res['TrainTestCutoffDate'] = train_test_cutoff
res['Train Size'] = 12
res['Test Size'] = 6
res['hdim'] = 16
res['BatchSize'] = 3

res.to_pickle('../mlp_onehot_traintest_results-Oct25_1735.pkl')

# create dataframe of training/test actual and predicted values

pred_train, pred_test = pd.DataFrame.from_dict(preds_train, orient='index'), pd.DataFrame.from_dict(preds_test, orient='index')
pred_train['Type'] = 'Train'
pred_test['Type'] = 'Test'
pred = pd.concat([pred_train, pred_test])
pred = pred.reset_index().rename(columns={'index': 'zcode'})

pred.to_pickle('../mlp_onehot_ypred-Oct25_1735.pkl')


# check 
checkpoint = torch.load("./model_info.pt")
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(epoch)
print(loss)

    
# plot training loss
plt.plot(res_train['loss_mse'])
# plot test loss
plt.plot(res_test['loss_mse'])

# plot mape
plt.plot(res_test['mape'])

mean_mape_train = res_train.groupby('epoch')['mape'].agg('mean')
mean_mape_test = res_test.groupby('epoch')['mape'].agg('mean')
plt.plot(mean_mape_train)
plt.plot(mean_mape_test)


mean_loss_train = res_train.groupby('epoch')['loss_mse'].agg('mean')
mean_loss_test = res_test.groupby('epoch')['loss_mse'].agg('mean')
plt.plot(mean_loss_train)
plt.plot(mean_loss_test)
