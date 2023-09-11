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

# get/create data from 1900 to 2023 (period in question) for every region/country
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

# 1. fill yield (tonnes/ha) based on whether tonnes & hectares exist, but tonnes/ha doesn't
crop_stats_df['yield_tonnes_per_ha'] = crop_stats_df.apply(lambda row: 
    row['tonnes'] / row['hectares'] if (not pd.isna(row['tonnes']) and 
                                               not pd.isna(row['hectares']) and 
                                               row['tonnes'] != 0 and 
                                               row['hectares'] != 0 and
                                               pd.isna(row['yield_tonnes_per_ha'])) else 
                                               (0 if row['tonnes'] == 0 and row['hectares'] == 0 else row['yield_tonnes_per_ha']), axis=1
)

# replace all remaining zeros and empty entries with NaNs
crop_stats_df['yield_tonnes_per_ha'].replace(['', ' ', 0], np.nan, inplace=True)
crop_stats_df['region'].replace(['', ' ', 0], np.nan, inplace=True)
crop_stats_df['hectares'].replace(['', ' ', 0], np.nan, inplace=True)
crop_stats_df['tonnes'].replace(['', ' ', 0], np.nan, inplace=True)

# look at missing data percentages for relevant variables: yield, tonnes and hectares per country across regions

# calculate the median/sum values for variables of interest per country and crop for each year across regions
variables_to_compute = ['yield_tonnes_per_ha', 'tonnes', 'hectares']
new_variable_names = ['yield_per_country', 'tonnes_per_country', 'hectares_per_country']
metrics = ['median', 'sum', 'sum']
grouping_columns = ['year', 'crop', 'country']
crop_stats_df = gfs.compute_aggregations_and_merge(crop_stats_df, variables_to_compute, new_variable_names, metrics, grouping_columns)

# plot barplots of missing values for variables of interest grouped by country
columns_to_visualize = ['yield_per_country', 'tonnes_per_country', 'hectares_per_country']
gfs.plot_missing_data_percentage(crop_stats_df, columns_to_visualize, ['maize', 'wheat'], 'country', pathout_plots)

# plot barplots of missing values for variables of interest grouped by year
gfs.plot_missing_data_percentage(crop_stats_df, columns_to_visualize, ['maize', 'wheat'], 'year', pathout_plots)

pd.set_option('display.max_rows', None)
print(data_to_plot.groupby('year')['yield_tonnes_per_ha'].apply(lambda x: x.isna().mean()))
print(new_crop_stats_df.groupby('year').size())
#--------------------------------------------------------------------------
# get distributions of yield across years for each country
gfs.plot_histograms(crop_stats_df, 'yield_per_country', 'country', ['maize', 'wheat'], 5, pathout_plots)
#--------------------------------------------------------------------------
# get distributions of yield per country for each 5-year period
crop_stats_df['year_section'] = ((crop_stats_df['year'] - crop_stats_df['year'].min()) // 5) * 5
gfs.plot_histograms(crop_stats_df, 'yield_per_country', 'year_section', ['maize', 'wheat'], 5, pathout_plots)
#--------------------------------------------------------------------------
# get time-series of yield (tonnes/ha) for each country across regions
crops = ['maize', 'wheat']
gfs.plot_time_series_by_group(crop_stats_df, 'yield_tonnes_per_ha', pathout_plots, crops, group_var='country',
                              metric='median',num_rows=5)

# get time-series of global yield (median) for each year across countries
gfs.plot_time_series_by_group(crop_stats_df, 'yield_per_country', pathout_plots, crops, group_var=None,
                              metric='median',num_rows=1)

#--------------------------------------------------------------------------
# DATA CLEANING
#--------------------------------------------------------------------------
# handling missing values:
# 1. missing values at the start and end of the time period considered (1900 - 2023) shall not be replaced; 
# 2. for missing values in between, we interpolate those using linspace() where missing 
# data periods do not exceed 10 (otherwise datapoints remain NaN)  

# create a copy of the original DataFrame
new_crop_stats_df = crop_stats_df.copy()

print(new_crop_stats_df['yield_per_country'].isna().sum())

# Assuming you've already imported numpy as np and pandas as pd
# and your interpolate_series function is defined as before.

# Assuming you've already imported numpy as np and pandas as pd
# and your interpolate_series function is defined as before.
unique_countries = new_crop_stats_df['country'].unique().tolist()

for country in unique_countries:
    for crop in crops:
        mask = (new_crop_stats_df['country'] == country) & (new_crop_stats_df['crop'] == crop)

        if mask.sum() > 0:  # Ensure we have rows for this country-crop combination
            original_na_count = new_crop_stats_df.loc[mask, 'yield_per_country'].isna().sum()
            
            # If there are NaN values, attempt interpolation
            if original_na_count > 0:
                filled_series = gfs.interpolate_series(new_crop_stats_df.loc[mask, 'yield_per_country'].copy())
                new_na_count = filled_series.isna().sum()

                # Print to see if interpolation has been effective
                if original_na_count != new_na_count:
                    print(f"Country: {country}, Crop: {crop} - NaNs before: {original_na_count}, NaNs after: {new_na_count}")

                new_crop_stats_df.loc[mask, 'yield_per_country'] = filled_series

# check remaining NaNs
print(new_crop_stats_df['yield_per_country'].isna().sum())
#--------------------------------------------------------------------------
# get new time-series of yield (tonnes/ha) for each country across regions
gfs.plot_time_series_by_group(crop_stats_df, 'yield_tonnes_per_ha', pathout_plots, crops, group_var='country',
                              metric='median',num_rows=5)

# get new time-series of global yield (median) for each year across countries
gfs.plot_time_series_by_group(crop_stats_df, 'yield_per_country', pathout_plots, crops, group_var=None,
                              metric='median',num_rows=1)

#--------------------------------------------------------------------------
# AGGREGATION
#--------------------------------------------------------------------------
# following Andersen et al., include data according to whether countries collectively contribute to some
# threshold value of crop yield (here 95% of tonnes - should use tonnes rather than tonnes/ha)

# check what proportion (%) of tonnes is missing for which country/crop & plot this
df_missing_values = new_crop_stats_df.groupby(["country", "crop"]
).apply(lambda x: (pd.isnull(x["tonnes_per_country"])).mean()).reset_index()

sns.barplot(data=df_missing_values, y=0, x='country', hue="crop")
plt.xticks(rotation=45, ha='right')
plt.xlabel('country')
plt.ylabel('% of Missing Values')
plt.title('tonnes: percentage of missing values per country & crop')
plt.show()

# include country in analysis only if it contributes to 95% of crop yield (tonnes) in 2010
# (following Andersen et al.)
new_crop_stats_df_2010 = new_crop_stats_df[new_crop_stats_df['year'] == 2010]

# calculate total tonnes by country for 2010
total_tonnes_by_country = new_crop_stats_df_2010.groupby('country')['tonnes'].sum().sort_values(ascending=False)

# Calculate cumulative sum
cumsum_tonnes = total_tonnes_by_country.cumsum()

# Determine the 90% cutoff
cutoff_90_percent = total_tonnes_by_country.sum() * 0.95

# Get countries that make up the first 90%
top_countries = cumsum_tonnes[cumsum_tonnes <= cutoff_90_percent].index

# Filter the original dataframe
new_crop_stats_df = new_crop_stats_df[new_crop_stats_df['country'].isin(top_countries)]

# calculate the median/sum values for variables of interest per crop for each year across countries
variables_to_compute = ['yield_per_country']
new_variable_names = ['global_yield']
metrics = ['sum']
grouping_columns = ['year', 'crop']
new_crop_stats_df = gfs.compute_aggregations_and_merge(new_crop_stats_df, variables_to_compute, new_variable_names, metrics, grouping_columns)

#--------------------------------------------------------------------------
# YEARLY CHANGE
#--------------------------------------------------------------------------
# get yearly change in global crop yield (separate for wheat and maize)

yearly_yield_df = new_crop_stats_df.groupby(['year', 'crop'])['global_yield'].first().reset_index()
yearly_yield_df['yearly_change_global_yield'] = yearly_yield_df.groupby('crop')['global_yield'].diff()

new_crop_stats_df = pd.merge(new_crop_stats_df, yearly_yield_df[['year', 'crop', 'yearly_change_global_yield']], on=['year', 'crop'], how='left')

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_yield_df, x="year", y="yearly_change_global_yield", hue="crop", ci=95)

plt.title('Yearly change in global median of yield (tonnes/ha)')
plt.xlabel('Year')
plt.ylabel('Change in global yield (tonnes/ha)')
plt.legend(title="Crop", loc="upper left")  # Position the legend
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{pathout_plots}median_global_yield_change_over_years.png", bbox_inches='tight', dpi=300)
    
plt.show()








