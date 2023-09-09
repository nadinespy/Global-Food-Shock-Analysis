import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

main_directory = '/media/nadinespy/NewVolume1/applications/ALLFED/work_trial/ALLFED-Global-Food-Shock-Analysis/'
pathin_data = main_directory+r'data/'
pathout_plots = main_directory+r'results/plots/'

os.chdir(main_directory)
from src import global_food_shocks as gfs

# use ALLFED style for plots (probably change where it's located)
plt.style.use(main_directory+r'ALLFED.mplstyle')

#--------------------------------------------------------------------------
# LOAD & VIEW THE DATA
#--------------------------------------------------------------------------

# load excel workbook
workbook = openpyxl.load_workbook(pathin_data+r'food-twentieth-century-crop-statistics-1900-2017-xlsx.xlsx')

# iterate through each sheet to have a look at what worksheets and columns 
# per worksheet are involved, and have a first look into the data
for index, sheet_name in enumerate(workbook.sheetnames):
    
    # get the sheet by name
    worksheet = workbook[sheet_name]  

    # get the header row (1st or 2nd row - it's different for different worksheets)
    if index == 1:
        header_row = worksheet[1]
    else:
        header_row = worksheet[2]

    # extract column names
    column_names = [cell.value for cell in header_row] 

    # create dataframe and look into first rows
    df = pd.read_excel(pathin_data+r'food-twentieth-century-crop-statistics-1900-2017-xlsx.xlsx', sheet_name=sheet_name)  
    
    # print the sheet name, column names, and first rows of each worksheet
    print(f"sheet name: {sheet_name}")
    print("column names:", column_names)
    print("first rows of dataframe:")
    print(df.head())
    print()  # add a newline for clarity

#--------------------------------------------------------------------------
# MODIFY DATA (EXTRACT RELEVANT VARS & RENAME)
#--------------------------------------------------------------------------
# extract crop stats worksheet - all information I need is in there
crop_stats_df = pd.read_excel(pathin_data+r'food-twentieth-century-crop-statistics-1900-2017-xlsx.xlsx', sheet_name='CropStats')  

# rename columns to avoid problems in saving plots & for convenience
rename_mapping = {
    'admin0': 'country',
    'production (tonnes)': 'tonnes',
    'hectares (ha)': 'hectares',
    'yield(tonnes/ha)': 'yield_tonnes_per_ha',
    'admin1': 'region'
}
gfs.rename_columns(crop_stats_df, rename_mapping)

# extract only wheat and maize data
crop_mask = crop_stats_df['crop'].isin(['wheat', 'maize'])
crop_stats_df = crop_stats_df[crop_mask]

# get data from 1900 to 2023 for every region/country
start_year = 1900
end_year = 2023

# create a dataframe for the desired year range for every country
all_years_df = pd.DataFrame({
    'year': list(range(start_year, end_year + 1))
})

# list to collect dataframes after reindexing for each country
dfs = []

for country in crop_stats_df['country'].unique():
    country_df = crop_stats_df[crop_stats_df['country'] == country]
    
    # merge the country dataframe with the all_years dataframe
    merged_df = all_years_df.merge(country_df, on='year', how='left')
    
    # fill the 'country' column with the country name for the new rows
    merged_df['country'].fillna(country, inplace=True)
    
    # append to the dfs list
    dfs.append(merged_df)

# concatenate all dataframes in dfs
crop_stats_df = pd.concat(dfs)
#--------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
#--------------------------------------------------------------------------
# look at missing data percentages for relevant variables: 
# yield (tonnes/ha), production (tonnes), hectares (ha)

# plot barplots of missing values for variables of interest grouped by country
columns_to_visualize = ['tonnes', 'hectares', 'yield_tonnes_per_ha']
gfs.plot_missing_data_proportions(crop_stats_df, 'country', columns_to_visualize, pathout_plots=pathout_plots)

#--------------------------------------------------------------------------
# check where tonnes and hectars are given, but not tonnes/ha, 
# plot percentages

# create a new column to represent the condition: first two columns 
# (production (tonnes) & hectares (ha)) are not missing, and the third one 
# (yield (tonnes/ha)) is missing
crop_stats_df['ton_and_hec_not_missing'] = crop_stats_df.apply(
    lambda row: 1 if (not pd.isna(row['tonnes']) and
                      not pd.isna(row['hectares']) and
                      pd.isna(row['yield_tonnes_per_ha'])) else 0, axis=1
)

# group the data by country, calculate the percentage of data where two columns not missing and the third one missing
grouped = crop_stats_df.groupby('country')['ton_and_hec_not_missing'].mean() * 100

# create the bar plot
plt.figure(figsize=(10, 6))

# plot the percentage of data where tonnes & hectares are not missing and 
# tonnes/ha is missing
plt.bar(
    grouped.index,  # x-axis (countries)
    grouped.values,  # height of the bars
    color='blue',
    label='tonnes & hectares exist, but tonnes/ha missing'
)

# customize the plot
plt.title('percentage of data where tonnes & hectares exist, but tonnes/ha is missing by country')
plt.xlabel('country')
plt.ylabel('percentage (%)')
plt.ylim(0, 100)  
plt.xticks(rotation=45, ha='right')  
plt.legend()
plt.savefig(pathout_plots+r'distr_ton_hec_not_miss.png', bbox_inches='tight')  # 'bbox_inches' ensures labels are not cut off

plt.show()
#--------------------------------------------------------------------------
# get distributions of yield (tonnes/ha) across years for each country

gfs.plot_histograms(crop_stats_df, 'yield_tonnes_per_ha', 'country', 5, pathout_plots)
#--------------------------------------------------------------------------
# get distributions of yield (tonnes/ha) across countries for each 5-year period

# group years into 5-year sections
crop_stats_df['year_section'] = ((crop_stats_df['year'] - crop_stats_df['year'].min()) // 5) * 5
gfs.plot_histograms(crop_stats_df, 'yield_tonnes_per_ha', 'year_section', 5, pathout_plots)
#--------------------------------------------------------------------------
# get time-series of total yield (tonnes/ha) median for each year across countries

data_to_plot = crop_stats_df[['year', 'yield_tonnes_per_ha']].reset_index(drop=True)

# use seaborn's lineplot function
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_to_plot, x="year", y="yield_tonnes_per_ha", ci=95, estimator="median")

plt.title('global median of yield (tonnes/ha) over time with bootstrapped uncertainty')
plt.xlabel('year')
plt.ylabel('yield (tonnes/ha)')
plt.grid(True)
plt.tight_layout()
plt.savefig(pathout_plots+r'global_median_yield_tonnes_per_ha_years.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------
# get time-series of yield (tonnes/ha) for each country across regions

gfs.plot_time_series_by_group(crop_stats_df, 'country', 'yield_tonnes_per_ha', pathout_plots, num_rows=5)

#--------------------------------------------------------------------------
# DATA CLEANING
#--------------------------------------------------------------------------
# handling missing values:
# 1. divide tonnes by hectar where tonnes and hectares are existent,
# but tonnes/hectares aren't, then eliminate zeros - those are likely dirty data/incorrect, 
# as countries will not have suddenly had zero yield in a given year, if before and after there
# was non-zero crop yield
# 2. missing values at the start and end of the time period considered (1900 - 2023) shall not be 
# # replaced; for missing values in between, we interpolate those using linspace() where missing 
# data periods do not exceed 10 (otherwise datapoints remain NaN)  

# create a copy of the original DataFrame
new_crop_stats_df = crop_stats_df.copy()

# 1. fill yield (tonnes/ha) based on whether tonnes & hectares exist, but tonnes/ha doesn't
new_crop_stats_df['yield_tonnes_per_ha'] = new_crop_stats_df.apply(lambda row: 
    row['tonnes'] / row['hectares'] if (not pd.isna(row['tonnes']) and 
                                               not pd.isna(row['hectares']) and 
                                               row['tonnes'] != 0 and 
                                               row['hectares'] != 0 and
                                               pd.isna(row['yield_tonnes_per_ha'])) else 
                                               (0 if row['tonnes'] == 0 and row['hectares'] == 0 else row['yield_tonnes_per_ha']), axis=1
)
# get new missing values plot for yield (tonnes/ha)

columns_to_visualize = ['yield_tonnes_per_ha']
gfs.plot_missing_data_proportions(new_crop_stats_df, 'country', columns_to_visualize, pathout_plots=pathout_plots)

# replace all zeros and empty entries with NaNs
new_crop_stats_df['yield_tonnes_per_ha'].replace(['', ' ', 0], np.nan, inplace=True)
new_crop_stats_df['region'].replace(['', ' ', 0], np.nan, inplace=True)
new_crop_stats_df['hectares'].replace(['', ' ', 0], np.nan, inplace=True)
new_crop_stats_df['tonnes'].replace(['', ' ', 0], np.nan, inplace=True)
#--------------------------------------------------------------------------
# 2. interpolate the data

print(new_crop_stats_df['yield_tonnes_per_ha'].isna().sum())

# group by country and find the number of unique regions for each
unique_regions_per_country = new_crop_stats_df.groupby('country')['region'].nunique()

# get countries without regions
countries_without_regions = new_crop_stats_df[new_crop_stats_df['region'].isna()]['country'].unique()

# get countries with regions
countries_with_regions = unique_regions_per_country[unique_regions_per_country > 1].index.tolist()

# for countries without regions
for country in countries_without_regions:
    country_mask = new_crop_stats_df['country'] == country
    filled_series = gfs.interpolate_series(new_crop_stats_df.loc[country_mask, 'yield_tonnes_per_ha'])
    new_crop_stats_df.loc[country_mask, 'yield_tonnes_per_ha'] = filled_series

# for countries with regions
for country in countries_with_regions:
    country_df = new_crop_stats_df[new_crop_stats_df['country'] == country]

    # for regions with partial data
    for region, region_df in country_df.groupby('region'):
        if not region_df['yield_tonnes_per_ha'].isna().all():
            mask = (new_crop_stats_df['country'] == country) & (new_crop_stats_df['region'] == region)
            filled_series = gfs.interpolate_series(new_crop_stats_df.loc[mask, 'yield_tonnes_per_ha'])
            new_crop_stats_df.loc[mask, 'yield_tonnes_per_ha'] = filled_series

# check remaining NaNs
print(new_crop_stats_df['yield_tonnes_per_ha'].isna().sum())
#--------------------------------------------------------------------------
# get new time-series of yield (tonnes/ha) for each country across regions

new_crop_stats_df['yield_tonnes_per_ha'] = pd.to_numeric(new_crop_stats_df['yield_tonnes_per_ha'], errors='coerce')
new_crop_stats_df['country'] = new_crop_stats_df['country'].astype('category')
new_crop_stats_df['region'] = new_crop_stats_df['region'].astype('category')

gfs.plot_time_series_by_group(crop_stats_df, 'country', 'yield_tonnes_per_ha', pathout_plots, num_rows=5)
#--------------------------------------------------------------------------
# AGGREGATING/WEIGHTING DATA
#--------------------------------------------------------------------------
# strategy: every year will have a global amount of harvested area, and each country or region 
# will have a portion of it - this portion will be used to weight the yield data per country/region;
# sometimes harvested area is missing data - in that case, the portion of harvested area that 
# this country or region has w. r. t. the global amount of harvested across all regions and 
# countries *and* all years

# first fill hectares based on whether tonnes & yield (tonnes/ha) exist, but hectares doesn't
new_crop_stats_df['hectares'] = new_crop_stats_df.apply(lambda row: 
    row['tonnes'] / row['yield_tonnes_per_ha'] if (not pd.isna(row['tonnes']) and 
                                               not pd.isna(row['yield_tonnes_per_ha']) and 
                                               row['tonnes'] != 0 and 
                                               row['yield_tonnes_per_ha'] != 0 and
                                               pd.isna(row['hectares'])) else 
                                               (0 if row['tonnes'] == 0 and row['yield_tonnes_per_ha'] == 0 else row['hectares']), axis=1
)

# get time-series of harvested area (ha) for each country across regions
gfs.plot_time_series_by_group(crop_stats_df, 'country', 'hectares', pathout_plots, num_rows=5)

# calculate the total harvested area globally for each year
global_hectares_per_year = new_crop_stats_df.groupby('year')['hectares'].sum()

# calculate the total harvested area globally across all years
global_hectares_total = new_crop_stats_df['hectares'].sum()

# calculate the total harvested area for each country/region across all years
total_hectares_per_country = new_crop_stats_df.groupby(['country', 'region'])['hectares'].sum()

# calculate the proportion of each country/region's harvested area to the global total across all years
all_years_proportion = total_hectares_per_country / global_hectares_total

# adjusting the original calculation:
def calculate_hectares_proportion(row):
    if pd.isna(row['hectares']):
        return all_years_proportion.get((row['country'], row['region']), 0)
    else:
        return row['hectares'] / global_hectares_per_year[row['year']]

new_crop_stats_df['hectares_proportion'] = new_crop_stats_df.apply(calculate_hectares_proportion, axis=1)

# calculate the weighted yield for each country/region using the hectares proportion
new_crop_stats_df['weighted_yield_tonnes_per_ha'] = new_crop_stats_df['hectares_proportion'] * new_crop_stats_df['yield_tonnes_per_ha']

# get time-series of weighted yield (tonnes/ha) for each country across regions
gfs.plot_time_series_by_group(new_crop_stats_df, 'country', 'weighted_yield_tonnes_per_ha', pathout_plots, num_rows=5)
import matplotlib.pyplot as plt


# Reset the index to avoid issues with duplicate indices
data_to_plot = new_crop_stats_df[['year', 'weighted_yield_tonnes_per_ha']].reset_index(drop=True)

# Use seaborn's lineplot function
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_to_plot, x="year", y="weighted_yield_tonnes_per_ha", ci=95, estimator="median")

plt.title('global median of weighted yield (tonnes/ha) over time with bootstrapped uncertainty')
plt.xlabel('Year')
plt.ylabel('Yield (tonnes/ha)')
plt.grid(True)
plt.tight_layout()
plt.savefig(pathout_plots+r'global_median_weighted_yield_tonnes_per_ha_years.png', bbox_inches='tight', dpi=300)
plt.show()










