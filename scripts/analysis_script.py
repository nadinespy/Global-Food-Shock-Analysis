import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# specify list of relevant column names
columns_to_analyze = ['production (tonnes)', 'hectares (ha)', 'yield(tonnes/ha)']  # Replace with your column names

# loop through the columns and create bar plots
for column_name in columns_to_analyze:
    # group the data by country and calculate the proportions
    grouped = crop_stats_df.groupby('admin0')[column_name].agg(
        total_count='count',  # total count of values
        missing_count=lambda x: x.isna().sum(),  # count of missing values
    ).reset_index()

    # calculate proportions as percentages
    grouped['missing_proportion'] = (grouped['missing_count'] / grouped['total_count']) * 100

    # create the stacked bar chart
    plt.figure(figsize=(10, 6))

    # plot the missing value proportion as a red bar
    plt.bar(
        grouped['admin0'],  # x-axis (countries)
        grouped['missing_proportion'],  # height of the red bars
        color='red',
        label='Missing Values'
    )

    # customize the plot
    plt.title(f'Percentage of Missing Values for {column_name} by Country')
    plt.xlabel('Country')
    plt.ylabel('Percentage of Missing Values (%)')
    plt.ylim(0, 100)  # set the y-axis limits to represent percentages
    plt.legend()
    plt.xticks(rotation=45, ha='right') # rotate x-axis labels for better readability
    plt.savefig(pathout_plots+f'distr_missing_values_{column_name}.png', bbox_inches='tight')  
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
grouped = crop_stats_df.groupby('admin0')['ton_and_hec_not_missing'].mean() * 100

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
plt.xlabel('Country')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)  
plt.xticks(rotation=45, ha='right')  
plt.legend()
plt.savefig(pathout_plots+r'distr_ton_hec_not_missing.png', bbox_inches='tight')  # 'bbox_inches' ensures labels are not cut off

plt.show()
#--------------------------------------------------------------------------
# get distributions of yield (tonnes/ha) for each country (across years)

# specify relevant columns
country_column_name = 'admin0'
yield_column_name = 'yield(tonnes/ha)'

# get unique country values from the dataframe
unique_countries = crop_stats_df[country_column_name].unique()

# create list of dataframes, each containing data for a single country
country_dataframes = [crop_stats_df[crop_stats_df[country_column_name] == country] for country in unique_countries]

# create a 5x5 grid of subplots (there are 25 countries)
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each country and create a histogram on each subplot
for i, country in enumerate(unique_countries):
    # filter the data for the current country
    data_for_country = new_crop_stats_df[new_crop_stats_df[country_column_name] == country]
    
    # create a histogram on the current subplot
    sns.histplot(data=data_for_country, x=yield_column_name, ax=axes[i], kde=True)
    axes[i].set_title(f'{country}')
    axes[i].set_xlabel(yield_column_name)
    axes[i].set_ylabel('density/frequency')

# remove any remaining empty subplots
for j in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.gcf().savefig(pathout_plots+r'distr_yield_ton_per_hec_countries.png', bbox_inches='tight', dpi=300)  # 'bbox_inches' ensures labels are not cut off

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
    sns.histplot(data=data_for_year_section, x=yield_column_name, ax=axes[i], kde=True)
    axes[i].set_title(f'Years {year_section}-{year_section + 4}')
    axes[i].set_xlabel(yield_column_name)
    axes[i].set_ylabel('Density')

# remove any remaining empty subplots
for j in range(len(unique_year_sections), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.savefig(pathout_plots+r'distr_yield_ton_per_hec_years.png', bbox_inches='tight', dpi=300)

plt.show()

#--------------------------------------------------------------------------
# get time-series of total yield (tonnes/ha) for each year, across countries

# specify relevant column
year_column_name = 'year'

# Group the data by year and calculate the total yield for each year
total_yield_per_year = crop_stats_df.groupby(year_column_name)[yield_column_name].sum()

# Create a time-series plot
plt.figure(figsize=(12, 6))
plt.plot(total_yield_per_year.index, total_yield_per_year.values, marker='o', linestyle='-')

# Customize the plot
plt.title('total yield per year across countries')
plt.xlabel('year')
plt.ylabel('total yield (tonnes/ha)')
plt.grid(True)
plt.savefig(pathout_plots+r'time_total_yield_ton_per_hec.png', bbox_inches='tight', dpi=300)

plt.show()
#--------------------------------------------------------------------------
# get total yield (tonnes/ha) for each year, separated for countries

# create a 5x5 grid of subplots
num_rows = 5
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)

# flatten the axes array for easier indexing
axes = axes.flatten()

# loop through each country and create a time-series plot on each subplot
for i, country in enumerate(unique_countries):
    # filter the data for the current country
    data_for_country = crop_stats_df[crop_stats_df[country_column_name] == country]
    
    # group the data by year and calculate the total yield for each year
    total_yield_per_year = data_for_country.groupby(year_column_name)[yield_column_name].sum()
    
    # create a time-series plot on the current subplot
    total_yield_per_year.plot(ax=axes[i], marker='o', linestyle='-', legend=True)
    
    # set the title as the country name
    axes[i].set_title(country)
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Total Yield (tonnes)')
    axes[i].grid(True)

    # add x-labels to every subplot in the bottom row
    if i >= num_rows * (num_cols - 1):
        axes[i].set_xlabel('Year')

# remove any remaining empty subplots
for j in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[j])

# adjust the layout to prevent overlap
plt.tight_layout()
plt.savefig(pathout_plots+r'time_yield_ton_per_hec.png', bbox_inches='tight', dpi=300)

plt.show()
#--------------------------------------------------------------------------
# DATA CLEANING
#--------------------------------------------------------------------------

# handling missing values:
# 1. divide tonnes by hectar where tonnes and hectares are existent,
# but tonnes/hectars aren't,
# 2. to be continued... 

# create a copy of the original DataFrame
new_crop_stats_df = crop_stats_df.copy()

ton_column_name = 'production (tonnes)'
hec_column_name = 'hectares (ha)'
yield_column_name = 'yield(tonnes/ha)'

# fill the third column based on conditions in the copy
new_crop_stats_df[yield_column_name] = new_crop_stats_df.apply(lambda row: 
    row[ton_column_name] / row[hec_column_name] if (not pd.isna(row[ton_column_name]) and 
                                               not pd.isna(row[hec_column_name]) and 
                                               row[ton_column_name] != 0 and 
                                               row[hec_column_name] != 0 and
                                               pd.isna(row[yield_column_name])) else 
                                               (0 if row[ton_column_name] == 0 and row[hec_column_name] == 0 else row[yield_column_name]), axis=1
)

#--------------------------------------------------------------------------
# get new missing values plot

# group the data by country and calculate the proportions
grouped = new_crop_stats_df.groupby(country_column_name)[yield_column_name].agg(
    total_count='count',  # total count of values
    missing_count=lambda x: x.isna().sum(),  # count of missing values
).reset_index()

# calculate proportions as percentages
grouped['missing_proportion'] = (grouped['missing_count'] / grouped['total_count']) * 100

# create the stacked bar chart
plt.figure(figsize=(10, 6))

# plot the missing value proportion as a red bar
plt.bar(
    grouped[country_column_name],   # x-axis (countries)
    grouped['missing_proportion'],  # height of the red bars
    color='red',
    label='missing values'
)

# customize the plot
plt.title(f'percentage of missing values for {yield_column_name} by country')
plt.xlabel('country')
plt.ylabel('percentage of missing values (%)')
plt.ylim(0, 100)
plt.legend()
plt.xticks(rotation=45, ha='right') 
plt.savefig(pathout_plots+r'distr_missing_values_cleaned.png', bbox_inches='tight')  
plt.show()
#--------------------------------------------------------------------------



