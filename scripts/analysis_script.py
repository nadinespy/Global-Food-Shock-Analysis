import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

main_directory = '/media/nadinespy/NewVolume1/applications/ALLFED/work_trial/ALLFED-Global-Food-Shock-Analysis/'
pathin_data = main_directory+r'data/'
pathout_plots = main_directory+r'results/plots/'

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
# EXPLORATORY DATA ANALYSIS
#--------------------------------------------------------------------------
# look at missing data percentages for relevant variables: 
# yield (tonnes/ha), production (tonnes), hectares (ha)

# extract crop stats worksheet - all information I need is in there
crop_stats_df = pd.read_excel(pathin_data+r'food-twentieth-century-crop-statistics-1900-2017-xlsx.xlsx', sheet_name='CropStats')  

# rename columns to avoid problems in saving plots & for convenience
crop_stats_df.rename(columns={'admin0': 'country'}, inplace=True)
crop_stats_df.rename(columns={'production (tonnes)': 'tonnes'}, inplace=True)
crop_stats_df.rename(columns={'hectares (ha)': 'hectares'}, inplace=True)
crop_stats_df.rename(columns={'yield(tonnes/ha)': 'yield_tonnes_per_ha'}, inplace=True)
crop_stats_df.rename(columns={'admin1': 'region'}, inplace=True)

# specify list of relevant column names
columns_to_analyze = ['tonnes', 'hectares', 'yield_tonnes_per_ha']  

# loop through the columns and create bar plots
for column_name in columns_to_analyze:
    # group the data by country and calculate the proportions
    grouped = crop_stats_df.groupby('country')[column_name].agg(
        total_count='count',  # total count of values
        missing_count=lambda x: x.isna().sum(),  # count of missing values
    ).reset_index()

    # calculate proportions as percentages
    grouped['missing_proportion'] = (grouped['missing_count'] / grouped['total_count']) * 100

    # create the stacked bar chart
    plt.figure(figsize=(10, 6))

    # plot the missing value proportion as a red bar
    plt.bar(
        grouped['country'],  # x-axis (countries)
        grouped['missing_proportion'],  # height of the red bars
        color='red',
        label='missing values'
    )

    # customize the plot
    plt.title(f'percentage of missing values for {column_name} by country')
    plt.xlabel('country')
    plt.ylabel('percentage of missing values (%)')
    plt.ylim(0, 100)  # set the y-axis limits to represent percentages
    plt.legend()
    plt.xticks(rotation=45, ha='right') # rotate x-axis labels for better readability
    plt.savefig(pathout_plots+f'distr_miss_val_{column_name}.png', bbox_inches='tight')  
    plt.show()
#--------------------------------------------------------------------------
# check where tonnes and hectars are given, but not tonnes/ha, 
# plot percentages

# create a new column to represent the condition: first two columns 
# (production (tonnes) & hectares (ha)) are not missing, and the third one 
# (yield (tonnes/ha)) is missing
crop_stats_df['ton_and_hec_not_missing'] = crop_stats_df.apply(
    lambda row: 1 if (not pd.isna(row[columns_to_analyze[0]]) and
                      not pd.isna(row[columns_to_analyze[1]]) and
                      pd.isna(row[columns_to_analyze[2]])) else 0, axis=1
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
# get distributions of yield (tonnes/ha) for each country (across years)

# get unique country values from the dataframe
unique_countries = crop_stats_df['country'].unique()

# create list of dataframes, each containing data for a single country
country_dataframes = [crop_stats_df[crop_stats_df['country'] == country] for country in unique_countries]

# create a 5x5 grid of subplots (there are 25 countries)
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each country and create a histogram on each subplot
for i, country in enumerate(unique_countries):
    # filter the data for the current country
    data_for_country = crop_stats_df[crop_stats_df['country'] == country]
    
    # create a histogram on the current subplot
    sns.histplot(data=data_for_country, x='yield_tonnes_per_ha', ax=axes[i], kde=True)
    axes[i].set_title(f'{country}')
    axes[i].set_xlabel('yield (tonnes/ha)')
    axes[i].set_ylabel('density/frequency')

# remove any remaining empty subplots
for j in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.suptitle('distributions of yield (tonnes/ha) across years for each country ', y=1.02, fontsize=20)
plt.gcf().savefig(pathout_plots+r'distr_yield_tonnes_per_ha_countries.png', bbox_inches='tight', dpi=300)  # 'bbox_inches' ensures labels are not cut off

plt.show()
#--------------------------------------------------------------------------
# get distributions of yield (tonnes/ha) for each year (across countries)

# group years into 5-year sections
crop_stats_df['year_section'] = ((crop_stats_df['year'] - crop_stats_df['year'].min()) // 5) * 5

# get unique year section values from the DataFrame
unique_year_sections = crop_stats_df['year_section'].unique()

# create a grid of subplots based on the number of unique year sections
num_rows = 5
num_cols = (len(unique_year_sections) + num_rows - 1) // num_rows  # calculate the number of columns based on num_rows
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each year section and create a histogram on each subplot
for i, year_section in enumerate(unique_year_sections):
    # Filter the data for the current year section
    data_for_year_section = crop_stats_df[crop_stats_df['year_section'] == year_section]
    
    # create a histogram on the current subplot
    sns.histplot(data=data_for_year_section, x='yield_tonnes_per_ha', ax=axes[i], kde=True)
    axes[i].set_title(f'years {year_section}-{year_section + 4}')
    axes[i].set_xlabel('yield (tonnes/ha)')
    axes[i].set_ylabel('density')

# remove any remaining empty subplots
for j in range(len(unique_year_sections), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.suptitle('distributions of yield (tonnes/ha) across countries for each 5-year period', y=1.02, fontsize=20)
plt.savefig(pathout_plots+r'distr_yield_tonnes_per_ha_years.png', bbox_inches='tight', dpi=300)

plt.show()
#--------------------------------------------------------------------------
# get time-series of total yield (tonnes/ha) for each year, across countries (median)

# compute the median yield for each year across all countries and regions
median_yield_per_year = crop_stats_df.groupby('year')['yield_tonnes_per_ha'].median()

# create a single plot for the time-series
plt.figure(figsize=(10, 6))
median_yield_per_year.plot(marker='o', linestyle='-')
plt.title('median yield (tonnes/ha) over years across all countries and regions', fontsize=14)
plt.xlabel('year', fontsize=12)
plt.ylabel('median yield (tonnes/ha)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(pathout_plots + r'median_yield_tonnes_per_ha_years.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------
# get time-series of yield (tonnes/ha) for each country across regions

# create a 5x5 grid of subplots
num_rows = 5
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each country and create a time-series plot on each subplot
for i, country in enumerate(unique_countries):
    # filter the data for the current country
    data_for_country = crop_stats_df[crop_stats_df['country'] == country]
    
    # group the data by year and calculate the total yield for each year across regions (using median)
    median_yield_per_year = data_for_country.groupby('year')['yield_tonnes_per_ha'].median()

    # create a time-series plot on the current subplot
    median_yield_per_year.plot(ax=axes[i], marker='o', linestyle='-', legend=True)
    
    # set the title as the country name
    axes[i].set_title(country)
    axes[i].set_xlabel('year')
    axes[i].set_ylabel('median yield (tonnes/ha)')
    axes[i].grid(True)

    # add x-labels to every subplot in the bottom row
    if i >= num_rows * (num_cols - 1):
        axes[i].set_xlabel('year')

# remove any remaining empty subplots
for j in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.suptitle('time-series of yield (tonnes/ha) for each country across regions (median)', y=1.02, fontsize=20)
plt.savefig(pathout_plots+r'median_yield_tonnes_per_ha_years_countries.png', bbox_inches='tight', dpi=300)

plt.show()
#--------------------------------------------------------------------------
# get time-series only from 1900 - 2017 of yield (tonnes/ha) for each country across regions

start_year = 1900
end_year = 2023

# Create a dataframe for the desired year range for every country
all_years_df = pd.DataFrame({
    'year': list(range(start_year, end_year + 1))
})

# List to collect dataframes after reindexing for each country
dfs = []

for country in crop_stats_df['country'].unique():
    country_df = crop_stats_df[crop_stats_df['country'] == country]
    
    # Merge the country dataframe with the all_years dataframe
    merged_df = all_years_df.merge(country_df, on='year', how='left')
    
    # Fill the 'country' column with the country name for the new rows
    merged_df['country'].fillna(country, inplace=True)
    
    # Append to the dfs list
    dfs.append(merged_df)

# Concatenate all dataframes in dfs
crop_stats_df = pd.concat(dfs)

# create a 5x5 grid of subplots
num_rows = 5
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each country and create a time-series plot on each subplot
for i, country in enumerate(unique_countries):
    # filter the data for the current country
    data_for_country = crop_stats_df[crop_stats_df['country'] == country]
    
    # group the data by year and calculate the total yield for each year across regions (using median)
    median_yield_per_year = data_for_country.groupby('year')['yield_tonnes_per_ha'].median()

    # create a time-series plot on the current subplot
    median_yield_per_year.plot(ax=axes[i], marker='o', linestyle='-', legend=True)
    
    # set the title as the country name
    axes[i].set_title(country)
    axes[i].set_xlabel('year')
    axes[i].set_ylabel('median yield (tonnes/ha)')
    axes[i].grid(True)

    # add x-labels to every subplot in the bottom row
    if i >= num_rows * (num_cols - 1):
        axes[i].set_xlabel('year')

# remove any remaining empty subplots
for j in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.suptitle('time-series (1900 - 2017) of yield (tonnes/ha) for each country across regions (median)', y=1.02, fontsize=20)
plt.savefig(pathout_plots+r'median_yield_tonnes_per_ha_years1900_2017_countries.png', bbox_inches='tight', dpi=300)

plt.show()
#--------------------------------------------------------------------------
# DATA CLEANING
#--------------------------------------------------------------------------

# handling missing values:
# 1. divide tonnes by hectar where tonnes and hectares are existent,
# but tonnes/hectares aren't,
# 2. eliminate zeros - those are likely dirty data/incorrect, as countries will 
# # not have suddenly had zero yield in a given year, if before and after there
# was non-zero crop yield

# create a copy of the original DataFrame
new_crop_stats_df = crop_stats_df.copy()

# fill yield (tonnes/ha) based on whether tonnes & hectares exist, but tonnes/ha doesn't
new_crop_stats_df['yield_tonnes_per_ha'] = new_crop_stats_df.apply(lambda row: 
    row['tonnes'] / row['hectares'] if (not pd.isna(row['tonnes']) and 
                                               not pd.isna(row['hectares']) and 
                                               row['tonnes'] != 0 and 
                                               row['hectares'] != 0 and
                                               pd.isna(row['yield_tonnes_per_ha'])) else 
                                               (0 if row['tonnes'] == 0 and row['hectares'] == 0 else row['yield_tonnes_per_ha']), axis=1
)
#--------------------------------------------------------------------------
# get new missing values plot

# group the data by country and calculate the proportions
grouped = new_crop_stats_df.groupby('country')['yield_tonnes_per_ha'].agg(
    total_count='count',  # total count of values
    missing_count=lambda x: x.isna().sum(),  # count of missing values
).reset_index()

# calculate proportions as percentages
grouped['missing_proportion'] = (grouped['missing_count'] / grouped['total_count']) * 100

# create the stacked bar chart
plt.figure(figsize=(10, 6))

# plot the missing value proportion as a red bar
plt.bar(
    grouped['country'],   # x-axis (countries)
    grouped['missing_proportion'],  # height of the red bars
    color='red',
    label='missing values'
)

# customize the plot
plt.title('percentage of missing values for yield (tonnes/ha) by country after some cleaning')
plt.xlabel('country')
plt.ylabel('percentage of missing values (%)')
plt.ylim(0, 100)
plt.legend()
plt.xticks(rotation=45, ha='right') 
plt.savefig(pathout_plots+r'distr_miss_val_yield_tonnes_per_ha_new.png', bbox_inches='tight')  
plt.show()
#--------------------------------------------------------------------------
# interpolate & pad the data

# replace all zeros and empty entries with NaNs
new_crop_stats_df['yield_tonnes_per_ha'].replace(['', ' ', 0], np.nan, inplace=True)
new_crop_stats_df['region'].replace(['', ' ', 0], np.nan, inplace=True)

# Convert 'yield_tonnes_per_ha' column to float to ensure correct operations with NaN
#new_crop_stats_df['yield_tonnes_per_ha'] = new_crop_stats_df['yield_tonnes_per_ha'].astype(float)

# Now, you can interpolate and pad
#unique_pairs = new_crop_stats_df[['country', 'region']].drop_duplicates().values

def fill_missing_values(series):
    # First, backward and forward padding
    series = series.fillna(method='bfill').fillna(method='ffill')
    
    # Next, fill with a moving average
    ma_series = series.rolling(3, center=True, min_periods=1).mean()
    series = series.fillna(ma_series)
    
    # Lastly, polynomial (quadratic) interpolation
    series = series.interpolate(method='polynomial', order=2)
    
    return series

# For countries without regions
for country in countries_without_regions:
    country_mask = new_crop_stats_df['country'] == country
    new_crop_stats_df.loc[country_mask, 'yield_tonnes_per_ha'] = fill_missing_values(new_crop_stats_df.loc[country_mask, 'yield_tonnes_per_ha'])

# For countries with regions
for country in countries_with_regions:
    country_df = new_crop_stats_df[new_crop_stats_df['country'] == country]
    
    # If the entire region is NaN, use country data
    for region, region_df in country_df.groupby('region'):
        if region_df['yield_tonnes_per_ha'].isna().all():
            median_yield = country_df[country_df['region'] != region]['yield_tonnes_per_ha'].median()
            mask = (new_crop_stats_df['country'] == country) & (new_crop_stats_df['region'] == region)
            new_crop_stats_df.loc[mask, 'yield_tonnes_per_ha'] = median_yield

    # For regions with partial data
    for region, region_df in country_df.groupby('region'):
        if not region_df['yield_tonnes_per_ha'].isna().all():
            mask = (new_crop_stats_df['country'] == country) & (new_crop_stats_df['region'] == region)
            filled_series = fill_missing_values(new_crop_stats_df.loc[mask, 'yield_tonnes_per_ha'])
            new_crop_stats_df.loc[mask, 'yield_tonnes_per_ha'] = filled_series

# Check remaining NaNs
print(new_crop_stats_df['yield_tonnes_per_ha'].isna().sum())
#--------------------------------------------------------------------------
# get new time-series of yield (tonnes/ha) for each country across regions

new_crop_stats_df['yield_tonnes_per_ha'] = pd.to_numeric(new_crop_stats_df['yield_tonnes_per_ha'], errors='coerce')
new_crop_stats_df['country'] = new_crop_stats_df['country'].astype('category')
new_crop_stats_df['region'] = new_crop_stats_df['region'].astype('category')

# create a 5x5 grid of subplots
num_rows = 5
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each country and create a time-series plot on each subplot
for i, country in enumerate(unique_countries):
    # filter the data for the current country
    data_for_country = new_crop_stats_df[new_crop_stats_df['country'] == country]
    
    # group the data by year and calculate the total yield for each year across regions (using median)
    median_yield_per_year = data_for_country.groupby('year')['yield_tonnes_per_ha'].median()

    # create a time-series plot on the current subplot
    median_yield_per_year.plot(ax=axes[i], marker='o', linestyle='-', legend=True)
    
    # set the title as the country name
    axes[i].set_title(country)
    axes[i].set_xlabel('year')
    axes[i].set_ylabel('median yield (tonnes/ha)')
    axes[i].grid(True)

    # add x-labels to every subplot in the bottom row
    if i >= num_rows * (num_cols - 1):
        axes[i].set_xlabel('year')

# remove any remaining empty subplots
for j in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.suptitle('time-series of yield (tonnes/ha) for each country across regions (median) after some cleaning', 
             y=1.02, fontsize=20)
plt.savefig(pathout_plots+r'median_yield_tonnes_per_ha_years1900_2017_countries_new.png', bbox_inches='tight', dpi=300)

plt.show()



