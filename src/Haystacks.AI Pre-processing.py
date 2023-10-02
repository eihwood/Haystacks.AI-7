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
with open("../data/SFR/rpi_index.pkl", "rb") as f:
    rpi = pickle.load(f)

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

# merge zipcode into mfr_rent and mfr_occ
mfr_occ = mfr_occ.merge(mfr_prop[["PID", "zipcode", "nounits"]], on="PID")
mfr_rent = mfr_rent.merge(mfr_prop[["PID", "zipcode"]], on="PID")


######################### CHECK MISSINGNESS #######################
# Use datetime.to_period() method to extract month and year
rpi["Month_Year"] = rpi["date"].dt.to_period("M")

print(
    str(rpi["Month_Year"].nunique()) + " Monthly Periods 2010-Present"
)  # 164 periods, monthly from 2010 to present

print(
    "Dates from: " + str(min(rpi["Month_Year"])) + " to " + str(max(rpi["Month_Year"]))
)

print(rpi["rental_index"].describe())

print(rpi["census_zcta5_geoid"].dtype)
rpi["census_zcta5_geoid"] = rpi["census_zcta5_geoid"].astype("int64")
# rpi.head()
# Subset on desired market zip codes using all of the zip codes in meta for desired markets
rpi = rpi[rpi["census_zcta5_geoid"].isin(meta["zipcode"].unique())]


# In[5]:


print(rpi["census_zcta5_geoid"].nunique())
print(meta["zipcode"].nunique())


# In[6]:


print(rpi["census_zcta5_geoid"].nunique() == meta["zipcode"].nunique())


# Extract unique zc from rpi
unique_zipcodes_rpi = set(rpi["census_zcta5_geoid"].unique())

# Extract unique zc from meta data
unique_zipcodes_meta = set(meta["zipcode"].unique())

# Check if unique values in df1 are in df2
if unique_zipcodes_meta.issubset(unique_zipcodes_rpi):
    print(
        "All zipcode values in the metadata markets for Atlanta and Cleveland are in RPI."
    )
else:
    print("Not all zipcode values in metadata are in RPI.")

res = list(set(meta["zipcode"]).difference(rpi["census_zcta5_geoid"]))
print(res)

missing_areas = meta[meta["zipcode"].isin(res)]
missing_areas = missing_areas[["Market", "Submarket", "zipcode"]].drop_duplicates(
    subset=["zipcode"]
)
missing_areas.to_csv("../data/SFR/SFR_RPI_missing_market_zipcodes.csv")
rpi.to_csv("../data/SFR/SFR_RPI_targetmarket_subset.csv")


# ### CHECK FOR DATA MISSINGNESS

# In[7]:


# Check for months missingness

# Set index to date, then resample by months and compute size of each group
s = rpi.set_index("date").resample("MS").size()
print(s[s == 0].index.tolist())  # no missing month-years

# The size of each date's' corresponds to the number of zipcodes for which there are rent index data in that year-month
s


# In[8]:


# Look for missingness
df = rpi.groupby(["census_zcta5_geoid"]).count()
print("N unique census_cbsa = " + str(len(df)))
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

rpi.groupby(["date"]).count()  # 11241 zip codes with rental indices


# ### Read in Haystacks.AI Housing Price Index
#

# In[9]:


hpi_base = open("../data/SFR/hpi_base.pkl", "rb")
hpi = pd.read_pickle(hpi_base)
hpi = pd.DataFrame(hpi)

hpi.head(200)
# Use datetime.to_period() method to extract month and year
hpi["Month_Year"] = hpi["period_start"].dt.to_period("M")

print(
    str(hpi["Month_Year"].nunique()) + " Monthly Periods 2007-Present"
)  # 164 periods, monthly from 2010 to present

print(
    "Dates from: " + str(min(hpi["Month_Year"])) + " to " + str(max(hpi["Month_Year"]))
)

print(hpi["price_index"].describe())


# In[10]:


print(hpi["census_zcta5_geoid"].dtype)
hpi["census_zcta5_geoid"] = hpi["census_zcta5_geoid"].astype("int64")

# Subset on desired market zip codes using all of the zip codes in meta for desired markets
hpi = hpi[hpi["census_zcta5_geoid"].isin(meta["zipcode"].unique())]

hpi.head()


# In[11]:


print(hpi["census_zcta5_geoid"].nunique() == meta["zipcode"].nunique())

print(hpi["census_zcta5_geoid"].nunique())
print(meta["zipcode"].nunique())
# Extract unique zc from rpi
unique_zipcodes_hpi = set(hpi["census_zcta5_geoid"].unique())

# Check if unique values in df1 are in df2
if unique_zipcodes_meta.issubset(unique_zipcodes_hpi):
    print(
        "All zipcode values in the metadata markets for Atlanta and Cleveland are in HPI."
    )
else:
    print("Not all zipcode values in metadata are in HPI.")

res = list(set(meta["zipcode"]).difference(hpi["census_zcta5_geoid"]))
print(res)

missing_areas = meta[meta["zipcode"].isin(res)]
missing_areas = missing_areas[["Market", "Submarket", "zipcode"]].drop_duplicates(
    subset=["zipcode"]
)
missing_areas.to_csv("../data/SFR/SFR_HPI_missing_market_zipcodes.csv")
hpi.to_csv("../data/SFR/SFR_HPI_targetmarket_subset.csv")


# In[12]:


# # Check if HPI and RPI have same subset of zipcodes
if unique_zipcodes_rpi.issubset(unique_zipcodes_hpi):
    print("All zipcode values in the RPI markets for Atlanta and Cleveland are in HPI.")
else:
    print("Not all zipcode values in RPI are in HPI.")

res = list(set(rpi["census_zcta5_geoid"]).difference(hpi["census_zcta5_geoid"]))
print(res)


# In[13]:


hpi.head()


# In[14]:


# Merge HPI and RPI

sfr = pd.merge(hpi, rpi, how="left", on=["census_zcta5_geoid", "Month_Year"])


# In[15]:


print(len(hpi))
print(len(rpi))
len(sfr) == len(hpi)


# In[16]:


# Look for missingness
df = sfr.groupby(["census_zcta5_geoid"]).count()
print("N unique zipcodes = " + str(len(df)))
# Test whether the values in rental index always equal date
result = (df["period_start"] == df["rental_index"]).all()

if result:
    print(
        "All counts (non NaN values) in date equal the counts in rental_index. No missing values"
    )
else:
    print(
        "Not all counts in date equal the counts in Rental Price Index indicating missing values."
    )

result = (df["period_start"] == df["price_index"]).all()

if result:
    print(
        "All counts (non NaN values) in date equal the counts in housing price_index. No missing values"
    )
else:
    print(
        "Not all counts in date equal the counts in House Price Index indicating missing values."
    )

sfr.groupby(["period_start"]).count()  # 11241 zip codes with rental indices

print(
    "rental price index starts in 2010 while housing price index starts in 2007, \
eliminate dates where there is no RPI as this is our target variable"
)

sfr.rename(columns={"census_cbsa_geoid_x": "census_cbsa_geoid"}, inplace=True)

sfr = sfr.dropna(subset=["rental_index"]).drop(
    columns=["census_cbsa_geoid_y", "period_start", "period_end", "trans_period"]
)


# In[17]:


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


# In[18]:


sfr.head()


# # Read in Multi-Family Data

# In[19]:


# Read in data
mfr_occ = pd.read_csv("../data/MFR/haystacks_occfile_7-26-2023.csv")
mfr_rent = pd.read_csv("../data/MFR/haystacks_rent_7-17-2023.csv")


# In[20]:


# Convert dates to datetime objects
mfr_rent["Period"] = pd.to_datetime(mfr_rent["Period"], format="%Y-%m-%d")
mfr_occ["Period"] = pd.to_datetime(mfr_occ["Period"], format="%m/%d/%Y")


# In[21]:


print(mfr_occ.head())
print(mfr_rent.head())


# In[22]:


print(len(mfr_occ) == len(mfr_rent))
print(len(mfr_occ))
print(len(mfr_rent))

mfr = pd.merge(
    mfr_rent, mfr_occ, how="left", left_on=["PID", "Period"], right_on=["PID", "Period"]
)


# In[23]:


# Subset for relevant PID for Atlanta and Cleveland Markets
pid_set = set(meta["PID"].unique())

mfr = mfr[mfr["PID"].isin(pid_set)]


# In[24]:


# Merge with meta data to get the zipcodes / market info

mfr = pd.merge(
    mfr,
    meta.drop(columns=["Occupancy", "OccupancyDate"]),
    left_on="PID",
    right_on="PID",
    how="left",
)

mfr.head()


# In[25]:


# Use datetime.to_period() method to extract month and year
mfr["Month_Year"] = mfr["Period"].dt.to_period("M")

print(
    str(mfr["Month_Year"].nunique()) + " Monthly Periods 2015-Present"
)  # 164 periods, monthly from 2010 to present

print(
    "Dates from: " + str(min(mfr["Month_Year"])) + " to " + str(max(mfr["Month_Year"]))
)

rent_desc_stats = mfr.groupby(["Market", "Submarket", "zipcode"])["Rent"].describe()

# print(mfr_rent_merge.groupby(['Market','Submarket'])['Occupancy'].describe().loc[['mean', 'std']])
rent_desc_stats.head()


# In[26]:


print(mfr["zipcode"].nunique() == meta["zipcode"].nunique())

print(mfr["zipcode"].nunique())
print(meta["zipcode"].nunique())
# Extract unique zc from rpi
unique_zipcodes_mfr = set(mfr["zipcode"].unique())

# Check if unique values in df1 are in df2
if unique_zipcodes_meta.issubset(unique_zipcodes_mfr):
    print(
        "All zipcode values in the metadata markets for Atlanta and Cleveland are in MFR."
    )
else:
    print("Not all zipcode values in metadata are in MFR.")

res = list(set(meta["zipcode"]).difference(mfr["zipcode"]))
print(res)

missing_areas = meta[meta["zipcode"].isin(res)]
missing_areas = missing_areas[["Market", "Submarket", "zipcode"]].drop_duplicates(
    subset=["zipcode"]
)
missing_areas.to_csv("../data/MFR/MFR_RentOcc_missing_market_zipcodes.csv")
mfr.to_csv("../data/MFR/MFR_RentOcc_targetmarket_subset.csv")


# ## Aggregate MFR by Month-Year and Zipcode
# - calculate median and mean standard deviation for rent price and occupancy
# - then calculate percentage change for median and mean

# In[27]:


mfr["median_rent"] = mfr.groupby(["zipcode", "Month_Year"])["Rent"].transform("median")
mfr["mean_rent"] = mfr.groupby(["zipcode", "Month_Year"])["Rent"].transform("mean")
mfr["std_rent"] = mfr.groupby(["zipcode", "Month_Year"])["Rent"].transform("std")

mfr["median_occ"] = mfr.groupby(["zipcode", "Month_Year"])["Occupancy"].transform(
    "median"
)
mfr["mean_occ"] = mfr.groupby(["zipcode", "Month_Year"])["Occupancy"].transform("mean")
mfr["std_occ"] = mfr.groupby(["zipcode", "Month_Year"])["Occupancy"].transform("std")


# In[28]:


# drop duplicates and just get the aggregated data
mfr.drop(
    columns=[
        "propertyname",
        "addressall",
        "state",
        "Longitude",
        "Latitude",
        "nounits",
        "Rent",
        "Occupancy",
        "UnitType",
        "PID",
        "Submarket",
    ],
    inplace=True,
)
mfr.drop_duplicates(subset=["Month_Year", "zipcode"], inplace=True)


# In[29]:


mfr.head()


# In[30]:


# Calculate percent change

# Make sure dataframe is in order
mfr.sort_values(["zipcode", "Month_Year"], inplace=True)

mfr["pc_mean_rent"] = mfr.groupby("zipcode")["mean_rent"].pct_change() * 100
mfr["pc_med_rent"] = mfr.groupby("zipcode")["median_rent"].pct_change() * 100
mfr["pc_mean_occ"] = mfr.groupby("zipcode")["mean_occ"].pct_change() * 100
mfr["pc_med_occ"] = mfr.groupby("zipcode")["median_occ"].pct_change() * 100

mfr.reset_index(inplace=True)


# In[31]:


mfr.drop(columns="index", inplace=True)
mfr.head()


# In[32]:


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
