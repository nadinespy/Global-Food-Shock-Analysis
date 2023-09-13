import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

main_directory = '/media/nadinespy/NewVolume1/applications/ALLFED/work_trial/ALLFED-Global-Food-Shock-Analysis/'
pathin_data = main_directory+r'data/'
pathout_plots = main_directory+r'results/plots/'

sys.path.append(main_directory)
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
# EXTRACT RELEVANT DATA SUBSET & RENAME VARIABLES
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

# extract only wheat & maize data
crops = ['maize', 'wheat']
crop_mask = crop_stats_df['crop'].isin(crops)
crop_stats_df = crop_stats_df[crop_mask]

# get/create data from 1900 to 2018 (period in question) for every region/country
# (NaNs for years where no data is available)
start_year = 1900
end_year = 2018

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
# divide tonnes by hectar where tonnes and hectares are existent, but tonnes/hectares aren't, 
# then eliminate zeros - those are likely dirty data/incorrect, as countries will not have suddenly 
# had zero yield in a given year, if before and after there was non-zero crop yield;
# likewise, calculate tonnes where hectares and tonnes/ha exist, but tonnes don't

# calculate tonnes where hectares and tonnes/ha exist, but tonnes don't
crop_stats_df = crop_stats_df.assign(tonnes=[row["tonnes"] 
                                         if not pd.isnull(row["tonnes"]) 
                                         else row["yield_tonnes_per_ha"]*row["hectares"] 
                                         for i, row in crop_stats_df.iterrows() ])

# calculate yield (tonnes/ha) where tonnes & hectares exist, but tonnes/ha doesn't
crop_stats_df['yield_tonnes_per_ha'] = crop_stats_df.apply(lambda row: 
    row['tonnes'] / row['hectares'] if (not pd.isna(row['tonnes']) and 
                                               not pd.isna(row['hectares']) and 
                                               row['tonnes'] != 0 and 
                                               row['hectares'] != 0 and
                                               pd.isna(row['yield_tonnes_per_ha'])) else 
                                               (0 if row['tonnes'] == 0 and row['hectares'] == 0 else row['yield_tonnes_per_ha']), axis=1
)

# replace all remaining zeros and empty entries with NaNs
columns_to_modify = ['yield_tonnes_per_ha', 'region', 'hectares', 'tonnes']
values_to_replace = ['', ' ', 0]
crop_stats_df = gfs.replace_values_in_columns(crop_stats_df, columns_to_modify, values_to_replace)

#--------------------------------------------------------------------------
# aggregate data across regions & look at missing data percentages for: 
# yield, tonnes & hectares

# calculate the median/sum values for variables of interest per country and crop for each year (i.e., across regions)
vars_to_aggregate = ['yield_tonnes_per_ha', 'tonnes', 'hectares']
aggregated_vars = ['yield_per_country', 'tonnes_per_country', 'hectares_per_country']
metrics = ['median', 'sum', 'sum']
grouping_vars = ['year', 'crop', 'country']
crop_stats_country_df = gfs.compute_aggregations(crop_stats_df, vars_to_aggregate, aggregated_vars, metrics, grouping_vars)

# replace all remaining zeros and empty entries with NaNs
crop_stats_country_df = gfs.replace_values_in_columns(crop_stats_country_df, aggregated_vars, values_to_replace)

# plot barplots of missing values for variables of interest grouped by country
miss_val_per_country = gfs.calc_miss_val_percentages(crop_stats_country_df, aggregated_vars, crops, ['country'])
gfs.plot_miss_val_percentages(miss_val_per_country, aggregated_vars, crops, ['country'], pathout_plots)

# plot barplots of missing values for variables of interest grouped by year
miss_val_per_year = gfs.calc_miss_val_percentages(crop_stats_country_df, aggregated_vars, crops, ['year'])
gfs.plot_miss_val_percentages(miss_val_per_year, aggregated_vars, crops, ['year'], pathout_plots)

#--------------------------------------------------------------------------
# get distributions of yield per country across years for each country
gfs.plot_histograms(crop_stats_country_df, 'yield_per_country', 'country', crops, 5, pathout_plots)
#--------------------------------------------------------------------------

# get distributions of yield per country for each 5-year period
crop_stats_country_df['year_section'] = ((crop_stats_country_df['year'] - crop_stats_country_df['year'].min()) // 5) * 5
gfs.plot_histograms(crop_stats_country_df, 'yield_per_country', 'year_section', crops, 5, pathout_plots)

#--------------------------------------------------------------------------
# get time-series of median yield (tonnes/ha) for each country across regions
gfs.plot_time_series_by_group(crop_stats_country_df, 'yield_per_country', pathout_plots, crops, group_var='country',
                              metric='median', num_rows=5)

#--------------------------------------------------------------------------
# DATA CLEANING
#--------------------------------------------------------------------------
# handling missing values:
# 1. missing values at the start and end of the time period considered (1900 - 2023) shall not be replaced; 
# 2. for missing values in between, we interpolate those using linspace() where missing 
# data periods do not exceed 10 (otherwise datapoints remain NaN)  

# create a copy of the original DataFrame
new_crop_stats_country_df = crop_stats_country_df.copy()

print(new_crop_stats_country_df['yield_per_country'].isna().sum())

# Assuming you've already imported numpy as np and pandas as pd
# and your interpolate_series function is defined as before.

# Assuming you've already imported numpy as np and pandas as pd
# and your interpolate_series function is defined as before.
unique_countries = new_crop_stats_country_df['country'].unique().tolist()

for country in unique_countries:
    for crop in crops:
        mask = (new_crop_stats_country_df['country'] == country) & (new_crop_stats_country_df['crop'] == crop)

        if mask.sum() > 0:  # Ensure we have rows for this country-crop combination
            original_na_count = new_crop_stats_country_df.loc[mask, 'yield_per_country'].isna().sum()
            
            # If there are NaN values, attempt interpolation
            if original_na_count > 0:
                filled_series = gfs.interpolate_series(new_crop_stats_country_df.loc[mask, 'yield_per_country'].copy())
                new_na_count = filled_series.isna().sum()

                # Print to see if interpolation has been effective
                if original_na_count != new_na_count:
                    print(f"Country: {country}, Crop: {crop} - NaNs before: {original_na_count}, NaNs after: {new_na_count}")

                new_crop_stats_country_df.loc[mask, 'yield_per_country'] = filled_series

# check remaining NaNs
print(new_crop_stats_country_df['yield_per_country'].isna().sum())

#--------------------------------------------------------------------------
# AGGREGATION
#--------------------------------------------------------------------------
# following Andersen et al., include data according to whether countries collectively contribute to some
# threshold value of crop yield (here 95% of tonnes - should use tonnes rather than tonnes/ha)

# include country in analysis only if it contributes to 95% of crop yield (tonnes) in 2010
# (following Andersen et al.)
new_crop_stats_country_df_2010 = new_crop_stats_country_df[new_crop_stats_country_df['year'] == 2010]

# calculate total tonnes by country for 2010
total_tonnes_by_country = new_crop_stats_country_df_2010.groupby('country')['tonnes_per_country'].sum().sort_values(ascending=False)

# Calculate cumulative sum
cumsum_tonnes = total_tonnes_by_country.cumsum()

# determine the 90% cutoff
cutoff_90_percent = total_tonnes_by_country.sum() * 0.95

# get countries that make up the first 90%
top_countries = cumsum_tonnes[cumsum_tonnes <= cutoff_90_percent].index

# filter the original dataframe
new_crop_stats_country_df = new_crop_stats_country_df[new_crop_stats_country_df['country'].isin(top_countries)]

# calculate the median/sum values for variables of interest per crop for each year across countries
# function doesn't work for some reason
variables_to_compute = ['yield_per_country']
new_variable_names = ['global_yield']
metrics = ['median']
grouping_columns = ['year', 'crop']
new_crop_stats_global_df = gfs.compute_aggregations(new_crop_stats_country_df, variables_to_compute, new_variable_names, metrics, grouping_columns)

# get new time-series of yield (tonnes/ha) for new list of countries
gfs.plot_time_series_by_group(new_crop_stats_country_df, 'yield_per_country', pathout_plots, crops, group_var='country',
                              metric='median',num_rows=5)

# get new time-series of yield (tonnes/ha) for new list of countries
gfs.plot_time_series_by_group(new_crop_stats_global_df, 'global_yield', pathout_plots, crops,
                              metric='median',num_rows=5)

#--------------------------------------------------------------------------
# YEARLY CHANGE
#--------------------------------------------------------------------------
# get yearly change in global crop yield (separate for wheat and maize)

#new_crop_stats_global_df = new_crop_stats_global_df.groupby(['year', 'crop'])['global_yield'].first().reset_index()
new_crop_stats_global_df['global_yield_change'] = new_crop_stats_global_df.sort_values("year", ascending=True).groupby('crop')['global_yield'].diff()
new_crop_stats_global_df = new_crop_stats_global_df[new_crop_stats_global_df['year'] != 1900]

# create a dictionary to map each crop to a unique index
crop_to_index = {crop: i for i, crop in enumerate(new_crop_stats_global_df['crop'].unique())}
color_map = plt.get_cmap()

plt.figure(figsize=(12, 6))

# calculate the lower and upper deciles for each crop
lower_deciles = new_crop_stats_global_df.groupby('crop')['global_yield_change'].quantile(0.1).to_dict()
upper_deciles = new_crop_stats_global_df.groupby('crop')['global_yield_change'].quantile(0.9).to_dict()

# plotting the main lineplot
sns.lineplot(data=new_crop_stats_global_df, x="year", y="global_yield_change", hue="crop", ci=95)

# adding horizontal lines and highlighting instances within the lower and upper deciles
dot_size = 50
for crop, decile in lower_deciles.items():
    plt.axhline(decile, color=color_map(crop_to_index[crop]), linestyle='--', label=f"{crop} lower decile")
    below_decile = new_crop_stats_global_df[(new_crop_stats_global_df['crop'] == crop) & (new_crop_stats_global_df['global_yield_change'] < decile)]
    plt.scatter(below_decile['year'], below_decile['global_yield_change'], color=color_map(crop_to_index[crop]), zorder=5, s=dot_size, label=f"{crop} below lower decile")
    
for crop, decile in upper_deciles.items():
    plt.axhline(decile, color=color_map(crop_to_index[crop]), linestyle='-.', label=f"{crop} upper decile")
    above_decile = new_crop_stats_global_df[(new_crop_stats_global_df['crop'] == crop) & (new_crop_stats_global_df['global_yield_change'] > decile)]
    plt.scatter(above_decile['year'], above_decile['global_yield_change'], color=color_map(crop_to_index[crop]), zorder=5, s=dot_size, marker="^", label=f"{crop} above upper decile")

plt.title('Yearly change in global median of yield (tonnes/ha)')
plt.xlabel('Year')
plt.ylabel('Change in global yield (tonnes/ha)')
plt.legend(title="Legend", loc="upper left", fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pathout_plots}median_global_yield_change_over_years.png", bbox_inches='tight', dpi=300)

plt.show()

#--------------------------------------------------------------------------
# distribution of global food decrease

# loop through each crop type
for crop in crops:
    subset = new_crop_stats_global_df[(new_crop_stats_global_df['crop'] == crop) & (new_crop_stats_global_df['global_yield_change'] < 0)]
    sns.histplot(data=subset, x="global_yield_change", label=crop, element="step", stat="density")

    plt.title('distribution of food decrease across all years')
    plt.xlabel('global yield decrease')
    plt.ylabel('Density')
    plt.legend(title="Crop")

    plt.savefig(f"{pathout_plots}distr_food_shocks_{crop}.png", bbox_inches='tight', dpi=300)
    plt.show()

#--------------------------------------------------------------------------
# TRENDS/PATTERNS/ANOMALIES
#--------------------------------------------------------------------------

# check for stationarity
result = adfuller(new_crop_stats_global_df['global_yield'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

#--------------------------------------------------------------------------
# decompose time-series
for crop in crops:
    subset_df = new_crop_stats_global_df[new_crop_stats_global_df['crop'] == crop]
    
    # ensure data is sorted by year for proper time-series decomposition
    subset_df = subset_df.sort_values('year')
    
    # decompose the time-series for the current crop
    result = seasonal_decompose(subset_df['global_yield'], model='additive', period=1)  # period=1 assumes no seasonality
    
    # plot the decomposition results manually
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    
    result.observed.plot(ax=ax1, title=f"{crop} - observed")
    result.trend.plot(ax=ax2, title=f"{crop} - trend")
    result.seasonal.plot(ax=ax3, title=f"{crop} - seasonal")
    result.resid.plot(ax=ax4, title=f"{crop} - residual")
    
    plt.tight_layout()
    plt.savefig(f"{pathout_plots}global_median_{crop}_decomposition.png", bbox_inches='tight', dpi=300)

    plt.show()

#--------------------------------------------------------------------------
# investigate trend using moving average
for crop in crops:
    subset_df = new_crop_stats_global_df[new_crop_stats_global_df['crop'] == crop]
    
    # ensure data is sorted by year for proper visualization
    subset_df = subset_df.sort_values('year')
    
    # calculate a rolling average as a simple method to visualize trend
    subset_df['rolling_avg'] = subset_df['global_yield'].rolling(window=5).mean()  # window size depends on your data's frequency and the trend's expected duration
    
    plt.figure(figsize=(12, 6))
    plt.plot(subset_df['year'], subset_df['global_yield'], label='Observed')
    plt.plot(subset_df['year'], subset_df['rolling_avg'], label='Rolling Average (Trend)', linestyle='--')
    plt.title(f"{crop} - observed vs. rolling average")
    plt.xlabel('year')
    plt.ylabel('global yield')
    plt.legend()
    plt.savefig(f"{pathout_plots}global_median_{crop}_moving_average.png", bbox_inches='tight', dpi=300)

    plt.show()

#--------------------------------------------------------------------------
# investigate autocorrelation
for crop in crops:
    # Assuming you've already filtered the dataframe for a specific crop
    subset_df = new_crop_stats_global_df[new_crop_stats_global_df['crop'] == crop]  # Replace 'maize' with the crop of interest

    # Drop any NaN values for accurate plotting
    subset_df = subset_df.dropna(subset=['global_yield'])

    # ACF and PACF plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    sm.graphics.tsa.plot_acf(subset_df['global_yield'].values.squeeze(), lags=40, ax=axes[0])
    sm.graphics.tsa.plot_pacf(subset_df['global_yield'].values.squeeze(), lags=40, ax=axes[1])
    plt.savefig(f"{pathout_plots}global_median_{crop}_autocorr.png", bbox_inches='tight', dpi=300)

    plt.show()

#--------------------------------------------------------------------------
# create world map with largest food decrease per country

# get yearly change in crop yield per country (separate for wheat and maize)
new_crop_stats_country_df['yearly_yield_change'] = new_crop_stats_country_df.groupby(['crop', 'country'])['yield_per_country'].diff()

# load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# loop through each crop type
for crop in crops: 

    # finding the largest decrease (food shock) for the specific crop
    largest_decreases = new_crop_stats_country_df[new_crop_stats_country_df['crop'] == crop].groupby('country')['yearly_yield_change'].min().reset_index()

    # merge the world map with the largest decreases data for the specific crop
    merged = world.set_index('name').join(largest_decreases.set_index('country'))

    # setup figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 25))

    # adjust colorbar size and add a label
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    merged.plot(column='yearly_yield_change', cmap='Reds_r', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, cax=cax)
    cax.set_ylabel('Largest Food Decrease')

    ax.set_title(f'Largest Food Decrease by Country for {crop.capitalize()}')
    plt.savefig(f"{pathout_plots}max_food_shock_{crop}.png", bbox_inches='tight', dpi=300)
    
    plt.show()









