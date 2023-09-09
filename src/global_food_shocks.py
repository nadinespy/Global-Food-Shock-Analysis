import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def rename_columns(df, rename_dict):
    """
    Renames columns of a DataFrame based on a provided dictionary.

    Args:
    - df (pd.DataFrame): The DataFrame whose columns are to be renamed.
    - rename_dict (dict): A dictionary where keys are the old column names and values are the new column names.

    Returns:
    - pd.DataFrame: DataFrame with renamed columns.
    """
    return df.rename(columns=rename_dict, inplace=True)


def plot_missing_data_proportions(df, grouping_var, columns_to_analyze, pathout_plots=''):
    """
    Plots the proportions of missing data for specified columns, grouped by a specified variable.

    Args:
    - df: DataFrame containing the data.
    - grouping_var: The variable/column name to group data by.
    - columns_to_analyze: List of column names to analyze and plot.
    - pathout_plots (optional): Path where the generated plots will be saved. 
                               If not provided, plots will just be shown and not saved.
    """
    for column_name in columns_to_analyze:
        # Group the data by the grouping variable and calculate the proportions
        grouped = df.groupby(grouping_var)[column_name].agg(
            total_count='count',  # Total count of values
            missing_count=lambda x: x.isna().sum(),  # Count of missing values
        ).reset_index()

        # Calculate proportions as percentages
        grouped['missing_proportion'] = (grouped['missing_count'] / grouped['total_count']) * 100

        # Create the stacked bar chart
        plt.figure(figsize=(10, 6))

        # Plot the missing value proportion as a red bar
        plt.bar(
            grouped[grouping_var],  # x-axis
            grouped['missing_proportion'],  # Height of the red bars
            color='red',
            label='missing values'
        )

        # Customize the plot
        plt.title(f'Percentage of missing values for {column_name} by {grouping_var}')
        plt.xlabel(grouping_var)
        plt.ylabel('Percentage of missing values (%)')
        plt.ylim(0, 100)  # Set the y-axis limits to represent percentages
        plt.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        if pathout_plots:
            plt.savefig(pathout_plots + f'distr_miss_val_{column_name}.png', bbox_inches='tight')
        plt.show()


def plot_histograms(df, plot_var, grouping_var, num_rows, pathout_plots):
    """
    Plot histograms of a variable across different groups using a grid of subplots.
    
    Parameters:
    - df (pandas.DataFrame): The input data frame containing the data.
    - plot_var (str): The name of the column in the dataframe to be plotted.
    - grouping_var (str): The name of the column used for grouping the data.
    - num_rows (int): Number of rows for the subplot grid.
    - pathout_plots (str): The path where the resulting plot should be saved.
    
    Returns:
    None. The function will save the plot to the specified path and display it.

    Example:
    >>> plot_histograms(crop_stats_df, 'yield_tonnes_per_ha', 'year_section', 5, './plots/')
    """
    # reset the DataFrame index
    df = df.reset_index(drop=True)

    # get unique values from the grouping variable
    unique_values = df[grouping_var].unique()

    # create a grid of subplots based on the number of unique values
    num_cols = (len(unique_values) + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # flatten the axes array for easier indexing
    axes = axes.flatten()

    # loop through each unique value and create a histogram on each subplot
    for i, value in enumerate(unique_values):
        # Filter the data for the current unique value
        data_for_value = df[df[grouping_var] == value]
        
        # create a histogram on the current subplot
        sns.histplot(data=data_for_value, x=plot_var, ax=axes[i], kde=True)
        axes[i].set_title(f'{value}')
        axes[i].set_xlabel(plot_var)
        axes[i].set_ylabel('density')

    # remove any remaining empty subplots
    for j in range(len(unique_values), len(axes)):
        fig.delaxes(axes[j])

    # adjust the layout to prevent overlap
    plt.tight_layout()
    plt.suptitle(f'distributions of {plot_var} across {grouping_var}', y=1.02, fontsize=20)
    plt.savefig(pathout_plots+f'distr_{plot_var}_{grouping_var}.png', bbox_inches='tight', dpi=300)

    plt.show()


def plot_time_series_by_group(df, group_var, y_var, pathout_plots, num_rows=5):
    """
    Plot a time series of a variable for each unique value in a grouping variable.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_var (str): The column name to be used for grouping (e.g., 'country').
    - y_var (str): The column name of the variable to be plotted (e.g., 'yield_tonnes_per_ha').
    - pathout_plots (str): Directory path where the plot will be saved.
    - num_rows (int): The number of rows for the grid of subplots. Default is 5.
    
    Returns:
    - None (The function will generate a grid of time-series plots, save it, and also display it.)
    """
    
    unique_groups = df[group_var].unique()
    
    # calculate the number of columns based on the given number of rows and unique groups
    num_cols = (len(unique_groups) + num_rows - 1) // num_rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)

    # flatten the axes array for easier indexing
    axes = axes.flatten()

    for i, group in enumerate(unique_groups):
        # filter the data for the current group
        data_for_group = df[df[group_var] == group]
        
        # group the data by year and calculate the median value for each year
        median_value_per_year = data_for_group.groupby('year')[y_var].median()

        # create a time-series plot on the current subplot
        median_value_per_year.plot(ax=axes[i], marker='o', linestyle='-')
        
        # set the title as the group name at the top
        axes[i].set_title(group, position=(0.5, 1.02))
        axes[i].set_xlabel('year')
        axes[i].set_ylabel(f'median {y_var}')
        axes[i].grid(True)

    # remove any remaining empty subplots
    for j in range(len(unique_groups), len(axes)):
        fig.delaxes(axes[j])

    # adjust the layout to prevent overlap and set a centered supertitle
    plt.tight_layout()
    fig.suptitle(f'time-series of {y_var} for each {group_var}', y=1.05, fontsize=20)
    plt.savefig(f"{pathout_plots}median_{y_var}_years_{group_var}.png", bbox_inches='tight', dpi=300)
    plt.show()

def interpolate_series(series):
    """
    Interpolates NaN sequences in a pandas Series using linear interpolation. 
    Only sequences of 1 to 10 NaNs surrounded by valid numbers are interpolated.

    Parameters:
    - series (pandas.Series): The series to interpolate.

    Returns:
    - pandas.Series: The interpolated series.

    Note:
    Sequences of NaNs at the start or end of the series are not interpolated.
    """
    # Helper function to handle NaN sequences
    def handle_sequence(start, end, series, interpolated_values):
        gap_size = end - start
        if 1 <= gap_size <= 10:
            # Ensure we're not on the edge of the series
            if start > 0 and end < len(series):
                linspace_values = np.linspace(series.iloc[start-1], series.iloc[end], gap_size+2)[1:-1]
                interpolated_values.iloc[start:end] = linspace_values

    # Placeholder for interpolated values
    interpolated_values = series.copy()

    # Manually loop to determine NaN sequences
    start = None
    for i in range(len(series)):
        if np.isnan(series.iloc[i]) and start is None:
            start = i
        elif start is not None and (not np.isnan(series.iloc[i]) or i == len(series)-1):
            handle_sequence(start, i, series, interpolated_values)
            start = None

    return interpolated_values

def check_for_duplicate_indices(df):
    """
    Checks a dataframe for duplicate indices and raises a ValueError if any are found.

    Parameters:
    - df (pandas.DataFrame): The dataframe to check.

    Raises:
    - ValueError: If duplicate indices are found.
    """
    duplicated = df.index[df.index.duplicated()].unique()
    if len(duplicated) > 0:
        raise ValueError(f"Duplicate indices found: {duplicated}")





