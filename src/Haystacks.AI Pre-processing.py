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
grouped = mfr_zc.groupby("zipcode")

# Step 2: Sort by 'Month_Year' within each group
sorted_df = grouped.apply(lambda group: group.sort_values(by="Period"))

# Step 3: Update the row index 0 of column 'pc_mean_rent' to be 100
sorted_df.loc[sorted_df.groupby(level="zipcode").head(1).index,["pc_mean_rent", "pc_mean_occ"],] = 100

# Step 4: Update any other NaN values to be 0
sorted_df.fillna(0, inplace=True)

# Create a new column 'cumulative_sum' that represents the cumulative sum within each group
sorted_df[["mfr_mean_rent_index","mfr_mean_occ_index"]] = (
    sorted_df[["pc_mean_rent", "pc_mean_occ"]]
    .groupby(level="zipcode")
    .cumsum())

# Step 3: Update the row index 0 of column 'pc_mean_rent' to be 100
sorted_df.loc[sorted_df.groupby(level="zipcode").head(1).index,
    ["mfr_mean_rent_index","mfr_mean_occ_index"]] = 100

sorted_df.head(200)

# Reset 'zipcode' as a regular column and drop some intermediate columns
sorted_df = sorted_df.droplevel("zipcode")
sorted_df.drop(columns = ['pc_mean_rent', 'pc_mean_occ'], inplace=True)

data = pd.merge(
    sfr,
    sorted_df,
    how="left",
    left_on=["date", "census_zcta5_geoid"],
    right_on=["Period", "zipcode"]).drop(columns = ['Month_Year', 'Period'])


data.head(100)



data.rename(
    columns={
        "rental_index": "sfr_rental_index",
        "price_index": "sfr_price_index",
        #"median_rent": "mfr_med_rent",
        "mean_rent_zc": "mfr_mean_rent",
        #"std_rent": "mfr_std_rent",
        #"median_occ": "mfr_med_occ",
        "occupancy": "mfr_occ",
        #"std_occ": "mfr_std_occ"
    }, inplace=True)

data.drop(
    columns="zipcode",
    inplace=True
)
data.head(200)




##################### MIGRATION DATA ##########################################

mig_ata = pd.read_csv("../data/Migration/area_migration_ga_zip.csv").assign(us_zip=lambda x: x['us_zip'].astype(str))
mig_clv = pd.read_csv("../data/Migration/haystacks-cleveland-migration-patterns.csv").assign(us_zip=lambda x: x['us_zip'].astype(str))

# Bind rows
if (mig_ata.columns == mig_clv.columns).all():
    mig = pd.concat([mig_ata, mig_clv], ignore_index=True)
else:
    print('Columns in migration Atlanta and Clevelend dataframes are not the same')
# Make date columns
mig['date'] = pd.to_datetime(mig['observation_start_date'])

# Subset on all_zips
mig = mig[mig["us_zip"].isin(all_zips)]

# Drop some unnecessary columns 
mig.drop(columns = ['area', 'location_id', 'us_state_id', 'us_zip_id', 'observation_start_date', 'observation_end_date'], inplace = True)

# Calculate population index (group by zipcode)
# Define a function to divide each element by the first element within each group
def divide_by_first_element(group):
    first_element = group.iloc[0]
    return group / first_element

# Use groupby and transform to apply the function within each group
mig.sort_values(['us_zip', 'date'], inplace = True)

# These don't change within a zipcode in our time series, so the indices all evaluate to 1
#mig['population_index'] = mig.groupby('us_zip')['population'].transform(divide_by_first_element)
#mig['student_pop_index'] = mig.groupby('us_zip')['student_population_fraction'].transform(divide_by_first_element)

mig['inflow_index'] = mig.groupby('us_zip')['inflow_estimated'].transform(divide_by_first_element)
mig['outflow_index'] = mig.groupby('us_zip')['outflow_estimated'].transform(divide_by_first_element)
mig['netflow_index'] = mig.groupby('us_zip')['netflow_estimated'].transform(divide_by_first_element)

mig['income_inflow_index'] = mig.groupby('us_zip')['median_income_inflow'].transform(divide_by_first_element)
mig['income_diff_index'] = mig.groupby('us_zip')['median_income_difference'].transform(divide_by_first_element)
mig['age_inflow_index'] = mig.groupby('us_zip')['median_age_inflow'].transform(divide_by_first_element)
mig['age_inflow_diff_index'] = mig.groupby('us_zip')['median_age_difference'].transform(divide_by_first_element)

# Drop Some Columns - not sure what the 'normalized columns' but I'm dropping them
mig = mig[['date', 'us_zip', 'population','student_population_fraction', 
       'netflow_estimated', 'inflow_estimated', 'outflow_estimated', 'cumulative_netflow_estimated',
       'median_income_inflow', 'median_income', 'median_income_difference',
       'median_age_inflow', 'median_age', 'median_age_difference',
       'inflow_index', 'outflow_index','netflow_index',
        'income_inflow_index', 'income_diff_index', 
        'age_inflow_index', 'age_inflow_diff_index']]
 # 'netflow_estimated_normalized', 'inflow_estimated_normalized','outflow_estimated_normalized',  'confidence_score'

# Set index to date, then resample by months and compute size of each group to check for missing monthly periods
s = mig.set_index("date").resample("MS").size()
print(s[s == 0].index.tolist())  # no missing month-years


# Look for data missingness
df = mig.groupby(["us_zip"]).count()
print("N unique zipcodes = " + str(len(df))) # 219 (missing one zipcode)

# Test whether the values in mig always equal date
result = (df["date"] == df["inflow_estimated"]).all()

if result:
    print(
        "All counts (non NaN values) in date equal the counts in rental_index. No missing values"
    )
else:
    print(
        "Not all counts in date equal the counts in Rental Price Index indicating missing values."
    )

# MERGE WITH MFR AND SFR DATA
data1 = pd.merge(data, mig, how = 'left', left_on = ['date', 'census_zcta5_geoid'], right_on = ['date', 'us_zip'])
data1.sort_values(['census_zcta5_geoid', 'date'], inplace=True)


# Write the DataFrame to a pickle file
data1.to_pickle('../data/sfr_mfr_mig_pre-processed.pkl')
