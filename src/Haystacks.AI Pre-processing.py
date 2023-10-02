#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


# # READ IN SFR, MFR, MIGRATION DATA
#
# ### Property File (meta data):
# A property snap shot that has the market by PID which is what we will use to map and extract the markets of interest
#
# ### SFR: Read in Haystacks.AI Rental Price Index and Housing Price Index
#
# Haystacks.AI generates their own unique zipcode-based rental price index over time \
# These data are available on a monthly time-step from Jan, 2010 to August, 2023
#
# ### Check RPI and HPI for data missingness then subset on target markets
# target markets: Atlanta and Cleveland Metropolitan areas (as id'd in the Property File)
#
# ### Read in MFR
# - Subset on Target Markets
# - Check for missingness
# - Explore how to aggregate
# - Aggregate and combine with SFR
#
# ### Read in Migration
######################################## LOAD DATA #######################################################
# Read in SFR
## RPI
with open("../data/SFR/rpi_index.pkl", "rb") as f:
    rpi = pickle.load(f)

## HPI
hpi_base = open("../data/SFR/hpi_base.pkl", "rb")
hpi = pd.read_pickle(hpi_base)
hpi = pd.DataFrame(hpi)

# read in MFR data
mfr_occ = pd.read_csv("../data/MFR/haystacks_occfile_7-26-2023.csv")
mfr_rent = pd.read_csv("../data/MFR/haystacks_rent_7-17-2023.csv")
mfr_prop = pd.read_csv("../data/PropertyFileAugust2023.csv")

# set correct data types for columns
mfr_prop["zipcode"] = mfr_prop["zipcode"].astype(str)
mfr_rent["Period"] = pd.to_datetime(mfr_rent["Period"])
mfr_occ["Period"] = pd.to_datetime(mfr_occ["Period"])
mfr_occ["Occupancy"] = mfr_occ["Occupancy"] / 100

# create lists of Atlanta/Cleveland zip codes
# define target cbsa for cleveland and atlanta area zipcodes
target_cbsa = ['12060', '17460']
rpi_zips = rpi.loc[rpi['census_cbsa_geoid'].isin(target_cbsa)]['census_zcta5_geoid'].unique().tolist()
hpi_zips = hpi.loc[hpi['census_cbsa_geoid'].isin(target_cbsa)]['census_zcta5_geoid'].unique().tolist()

hpi_zips == rpi_zips # TRUE so every zip in RPI is in HPI

# create lists of unique PIDs in mfr_prop for ATL/CLE
mfr_pids = (mfr_prop.loc[mfr_prop.zipcode.isin(rpi_zips)]["PID"].unique().tolist())

# subset mfr_occ and mfr_rent by markets
mfr_occ_markets = mfr_occ.loc[mfr_occ.PID.isin(mfr_pids)]
mfr_rent_markets = mfr_rent.loc[mfr_rent.PID.isin(mfr_pids)]

# merge each with zip code and get lists of unique zip codes
mfr_occ_zips = (mfr_occ_markets.merge(mfr_prop[["PID", "zipcode"]], on="PID")["zipcode"].unique().tolist())
mfr_rent_zips = (mfr_rent_markets.merge(mfr_prop[["PID", "zipcode"]], on="PID")["zipcode"].unique().tolist())

# create sets to find applicable zip codes; creates a list for each market of zip codes where we have ALL data for MFR and SFR
all_zips = list(set(mfr_occ_zips) & set(mfr_rent_zips))

# subset MFR data by applicable zip codes
mfr_pids = mfr_prop.loc[mfr_prop.zipcode.isin(all_zips)]["PID"].unique().tolist()
mfr_occ = mfr_occ.loc[mfr_occ.PID.isin(mfr_pids)]
mfr_rent = mfr_rent.loc[mfr_rent.PID.isin(mfr_pids)]

# subset SFR RPI and HPI data by applicable zip codes
rpi = rpi[rpi['census_zcta5_geoid'].isin(all_zips)]
hpi = hpi[hpi['census_zcta5_geoid'].isin(all_zips)]

# merge zipcode into mfr_rent and mfr_occ
mfr_occ = mfr_occ.merge(mfr_prop[["PID", "zipcode", "nounits"]], on="PID")
mfr_rent = mfr_rent.merge(mfr_prop[["PID", "zipcode"]], on="PID")

# merge mfr occupancy and mfr rent
set(mfr_rent['zipcode']) == set(mfr_occ['zipcode']) # TRUE
# NOTE FROM EIH - IF WE ONLY WANT PIDS WHERE WE HAVE BOTH OCCUPANCY AND RENT REMOVE THE 'how = 'left'' 
mfr = mfr_rent.merge(mfr_occ[['PID','Period','Occupancy', 'nounits']], on = ['PID','Period'], how = 'left')
len(mfr) == len(mfr_rent)
mfr.sort_values(['Period', 'PID'], inplace=True)

# Check
print(rpi["census_zcta5_geoid"].nunique() == mfr["zipcode"].nunique()) # TRUE!
print("Number of Unique Zipcodes in RPI for Atlanta and Cleveland Markets = ", str(rpi["census_zcta5_geoid"].nunique())) # 200
set(rpi['census_zcta5_geoid']) == set(hpi['census_zcta5_geoid']) == set(mfr['zipcode']) # TRUE!

rpi.groupby(["date", 'census_cbsa_geoid']).count()  # 220 zip codes with rental indices, 144 in Atlanta, 76 in Cleveland

######################### CHECK MISSINGNESS #######################
# Use datetime.to_period() method to extract month and year
rpi["Month_Year"] = rpi["date"].dt.to_period("M")
print(str(rpi["Month_Year"].nunique()) + " Monthly Periods 2010-Present for SFR RPI")  # 164 periods, monthly from 2010 to present
print("Dates from: " + str(min(rpi["Month_Year"])) + " to " + str(max(rpi["Month_Year"])) + "for SFR RPI")
print(rpi["rental_index"].describe())

hpi["Month_Year"] = hpi['period_start'].dt.to_period("M")
print(str(hpi["Month_Year"].nunique()) + " Monthly Periods 2007-Present for SFR HPI")  # 200 periods, monthly from 2010 to present
print("Dates from: " + str(min(hpi["Month_Year"])) + " to " + str(max(hpi["Month_Year"])) + "for SFR HPI")

mfr["Month_Year"] = mfr['Period'].dt.to_period("M")
print(str(mfr["Month_Year"].nunique()) + " Monthly Periods 2015-06/2023 for MFR Rent and Occupancy")  # 200 periods, monthly from 2010 to present
print("Dates from: " + str(min(mfr["Month_Year"])) + " to " + str(max(mfr["Month_Year"])) + "MFR Rent and Occupancy")
print(mfr["Occupancy"].describe())
print(mfr["Rent"].describe())

rpi.groupby(["date", 'census_cbsa_geoid']).count()  # 220 zip codes with rental indices, 144 in Atlanta, 76 in Cleveland
hpi.groupby(["period_start", 'census_cbsa_geoid']).count()  # 220 zip codes with rental indices, 144 in Atlanta, 76 in Cleveland
mfr.groupby(['Period', ])
## Check for months missingness
# Set index to date, then resample by months and compute size of each group
s = rpi.set_index("date").resample("MS").size()
print(s[s == 0].index.tolist())  # no missing month-years

s = hpi.set_index("period_start").resample("MS").size()
print(s[s == 0].index.tolist())  # no missing month-years


s = mfr.set_index("Period").resample("MS").size()
print(s[s == 0].index.tolist())  # no missing month-years in MFR

# NO MISSING MONTH PERIODS

######################## Merge HPI and RPI #########################

sfr = pd.merge(rpi, hpi[['price_index','Month_Year', 'census_zcta5_geoid', 'coef']], how="left", on=["census_zcta5_geoid", "Month_Year"])

len(sfr) == len(rpi) # TRUE, merged as expected

sfr.head(200)

# Look for missingness
df = sfr.groupby(["census_zcta5_geoid"]).count()
print("N unique zipcodes = " + str(len(df)))
# Test whether the values in rental index always equal date
result = (df["date"] == df["rental_index"]).all()

if result:
    print(
        "All counts (non NaN values) in date equal the counts in rental_index. No missing values"
    )
else:
    print(
        "Not all counts in date equal the counts in Rental Price Index indicating missing values."
    )

result = (df["date"] == df["price_index"]).all()

if result:
    print(
        "All counts (non NaN values) in date equal the counts in housing price_index. No missing values"
    )
else:
    print(
        "Not all counts in date equal the counts in House Price Index indicating missing values."
    )

sfr.groupby(["date"]).count()  # 220 zip codes with rental indices

print("rental price index starts in 2010 while housing price index starts in 2007, \
eliminate dates where there is no RPI as this is our target variable")

# Rearrange the dataframe columns
cols = [
    "Month_Year",
    "date",
    "census_cbsa_geoid",
    "census_zcta5_geoid",
    "rental_index",
    "price_index",
    "coef",
]
sfr = sfr[cols]


############################# Aggregate MFR by Month-Year and Zipcode #############################
# - calculate aggregate rent and occupancy by zipcode
# - then calculate percentage change as an index similar to RPI and HPI

# add column for occupied units (occupancy and nounits are both on the PID level, so we'll reduce the mfr dataframe
mfr_sub = mfr.drop(columns='UnitType').drop_duplicates(subset=['PID', 'Period'])

mfr_sub['occupied_units'] = mfr_sub['Occupancy'] * mfr_sub['nounits']

# group and aggregate
mfr_oi = mfr_sub.groupby(['zipcode', 'Period'], as_index=False)[['nounits','occupied_units']].sum()

# add column for occupancy by zip code
mfr_oi['occupancy'] = mfr_oi['occupied_units'] / mfr_oi['nounits']

# Calculate the mean rent for each building, regardless of unit BR#, then take the mean of all buildings in a given zipcode
# group and aggregate

mfr['mean_rent_PID'] = mfr.groupby(['PID', 'Period'])[['Rent']].transform('mean')
mfr_ri = mfr.drop(columns = ['UnitType', 'Rent']).drop_duplicates()

mfr_ri = mfr_ri.groupby(['zipcode', 'Period'], as_index=False)[['mean_rent_PID']].mean()
mfr_ri.rename(columns = {'mean_rent_PID': 'mean_rent_zc'}, inplace=True)
# merge
mfr_zc = pd.merge(mfr_oi, mfr_ri, on = ['Period', 'zipcode'])

# Calculate occ and rent indices as percent change

# Make sure dataframe is in order
mfr_zc.sort_values(["zipcode", "Period"], inplace=True)
mfr_zc["pc_mean_rent"] = mfr_zc.groupby("zipcode")["mean_rent_zc"].pct_change() * 100
mfr_zc["pc_mean_occ"] = mfr_zc.groupby("zipcode")["occupancy"].pct_change() * 100
mfr_zc.reset_index(inplace=True)
mfr_zc.drop(columns="index", inplace=True)

# Step 1: Group the DataFrame by 'zipcode'
grouped = mfr.groupby("zipcode")

# Step 2: Sort by 'Month_Year' within each group
sorted_df = grouped.apply(lambda group: group.sort_values(by="Month_Year"))

# Step 3: Update the row index 0 of column 'pc_mean_rent' to be 100
sorted_df.loc[
    sorted_df.groupby(level="zipcode").head(1).index,
    ["pc_mean_rent", "pc_med_rent", "pc_mean_occ", "pc_med_occ"],
] = 100

# Step 4: Update any other NaN values to be 0
sorted_df.fillna(0, inplace=True)


# In[33]:


sorted_df.head()


# In[34]:


# Create a new column 'cumulative_sum' that represents the cumulative sum within each group
sorted_df[
    [
        "mfr_mean_rent_index",
        "mfr_med_rent_index",
        "mfr_mean_occ_index",
        "mfr_med_occ_index",
    ]
] = (
    sorted_df[["pc_mean_rent", "pc_med_rent", "pc_mean_occ", "pc_med_occ"]]
    .groupby(level="zipcode")
    .cumsum()
)

# Step 3: Update the row index 0 of column 'pc_mean_rent' to be 100
sorted_df.loc[
    sorted_df.groupby(level="zipcode").head(1).index,
    [
        "mfr_mean_rent_index",
        "mfr_med_rent_index",
        "mfr_mean_occ_index",
        "mfr_med_occ_index",
    ],
] = 100


# In[35]:


sorted_df.head(200)


# In[36]:


# Reset 'zipcode' as a regular column
sorted_df = sorted_df.droplevel("zipcode")


# In[37]:


sorted_df.head()


# In[38]:


final = pd.merge(
    sfr,
    sorted_df,
    how="left",
    left_on=["Month_Year", "census_zcta5_geoid"],
    right_on=["Month_Year", "zipcode"],
)


# In[39]:


final.head(100)


# In[40]:


final.rename(
    columns={
        "rental_index": "sfr_rental_index",
        "price_index": "sfr_price_index",
        "median_rent": "mfr_med_rent",
        "mean_rent": "mfr_mean_rent",
        "std_rent": "mfr_std_rent",
        "median_occ": "mfr_med_occ",
        "mean_occ": "mfr_mean_occ",
        "std_occ": "mfr_std_occ",
        "mean_rent_index": "mfr_mean_rent_index",
        "med_rent_index": "mfr_med_rent_index",
        "mean_occ_index": "mfr_mean_occ_index",
        "med_occ_index": "mfr_med_occ_index",
    },
    inplace=True,
)


# In[41]:


final.drop(
    columns=[
        "Period",
        "date",
        "zipcode",
        "pc_mean_rent",
        "pc_med_rent",
        "pc_mean_occ",
        "pc_med_occ",
    ],
    inplace=True,
)


# In[42]:


final.head(200)


# In[44]:


final.to_csv("../data/SFRMFR_combined.csv")


# # Read in Migration Data

# In[59]:


mig_ata = pd.read_csv("../data/Migration/area_migration_ga_zip.csv")
mig_clv = pd.read_csv(
    "../data/Migration/haystacks_cleveland_market_tract_migration.csv"
)


# In[56]:


mig_ata = mig_ata[mig_ata["us_zip"].isin(meta["zipcode"])]
mig_clv = mig_clv[mig_clv["us_county_id"].isin(meta["zipcode"])]


# In[57]:


if set(meta["zipcode"]).issubset(set(mig_ata["us_zip"])):
    print("All zipcode values in the migration markets for Atlanta meta.")
else:
    print("Not all zipcode values in META are in HPI.")

res = list(set(mig_ata["us_zip"]).difference(meta["zipcode"]))
print(res)


# In[60]:


mig_clv.head()


# In[55]:


mig_ata.head()


# In[ ]:
