import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_missing_data_percentage(df, variables_to_plot, crops, grouping_variable, pathout):
    """
    Plots missing data percentage of the specified variables across different crops and years grouped by the specified variable.
    
    Parameters:
    - df: Pandas DataFrame
    - variables_to_plot: List of column names whose missing data percentage needs to be plotted
    - crops: List of crops to be plotted
    - grouping_variable: Variable by which data is to be grouped (e.g., country)
    - pathout: Output path for saving the plots (without file extension)
    """
    
    # Filter the dataframe for only the crops in question
    df = df[df['crop'].isin(crops)]

    for variable in variables_to_plot:
        # Calculate the missing data percentages
        if grouping_variable == 'year':
            missing_percentage = df.groupby(['year', 'crop']).apply(lambda group: group[variable].isna().mean() * 100).rename('missing_percentage').reset_index()
        else:
            grouping_columns = [grouping_variable, 'year', 'crop'] if isinstance(grouping_variable, str) else list(grouping_variable) + ['year', 'crop']
            missing_percentage = df.groupby(grouping_columns).apply(lambda group: group[variable].isna().mean() * 100).rename('missing_percentage').reset_index()

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.barplot(x=grouping_variable, y='missing_percentage', hue='crop', data=missing_percentage)
        plt.title(f"Missing Data Percentage for {variable} grouped by {grouping_variable}")
        plt.ylabel("Missing Percentage (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        

        # Construct output path for each variable
        #full_pathout = f"{pathout}_{variable}.png"
        #plt.savefig(pathout_plots+f'distr_{plot_var}_{grouping_var}.png', bbox_inches='tight', dpi=300)
        plt.savefig(pathout+f'distr_{variable}_by_{grouping_variable}.png', bbox_inches='tight', dpi=300)
        plt.show()
        #plt.savefig(full_pathout)
        plt.close()


def plot_histograms(df, plot_var, grouping_var, crops, num_rows, pathout_plots):
    """
    Plot histograms of a variable across different groups for specified crops using a grid of subplots.
    
    Parameters:
    - df (pandas.DataFrame): The input data frame containing the data.
    - plot_var (str): The name of the column in the dataframe to be plotted.
    - grouping_var (str): The name of the column used for grouping the data.
    - crops (list): List of crops for which histograms should be plotted.
    - num_rows (int): Number of rows for the subplot grid.
    - pathout_plots (str): The path where the resulting plot should be saved.
    
    Returns:
    None. The function will save the plot to the specified path and display it.
    """
    # reset the DataFrame index
    df = df.reset_index(drop=True)

    # filter dataframe to include only specified crops
    df = df[df['crop'].isin(crops)]

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
        
        # create a histogram on the current subplot, color-coded for each crop
        for crop in crops:
            sns.histplot(data=data_for_value[data_for_value['crop'] == crop], x=plot_var, ax=axes[i], kde=True, label=crop)
            axes[i].legend()
            axes[i].set_title(f'{value}')
            axes[i].set_xlabel(plot_var)
            axes[i].set_ylabel('density')

    # remove any remaining empty subplots
    for j in range(len(unique_values), len(axes)):
        fig.delaxes(axes[j])

    # adjust the layout to prevent overlap
    plt.tight_layout()
    plt.suptitle(f'distributions of {plot_var} across {grouping_var}', y=1.02, fontsize=20)
    plt.savefig(pathout_plots+f'distr_{plot_var}_across_{grouping_var}.png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_time_series_by_group(df, y_var, pathout_plots, crops, group_var=None, metric='median', num_rows=5):
    """
    Plot a time series of a variable for each unique value in a grouping variable using Seaborn's bootstrapping functionality.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - y_var (str): The column name of the variable to be plotted (e.g., 'yield_tonnes_per_ha').
    - pathout_plots (str): Directory path where the plot will be saved.
    - crops (list): List of crops to be plotted.
    - group_var (str, optional): The column name to be used for grouping (e.g., 'country'). If None, no grouping is done.
    - metric (str): The aggregation metric to use (e.g., 'median', 'sum'). Default is 'median'.
    - num_rows (int): The number of rows for the grid of subplots. Default is 5.
    
    Returns:
    - None (The function will generate a grid of time-series plots, save it, and also display it.)
    """
    
    if metric not in ['sum', 'median', 'mean']:
        raise ValueError("Invalid metric. Supported metrics are: 'sum', 'median', 'mean'")

    # Filter dataframe by the crops provided
    df = df[df['crop'].isin(crops)]
    
    # If no grouping is required
    if group_var is None:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="year", y=y_var, hue='crop', ci=95, estimator=metric)
        plt.title(f'Time Series of {metric} {y_var}', fontsize=20)
        plt.xlabel('Year')
        plt.ylabel(f'{metric} {y_var}')
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title='Crop')
        plt.savefig(f"{pathout_plots}{metric}_{y_var}_years.png", bbox_inches='tight', dpi=300)
        plt.show()
        return

    unique_groups = df[group_var].unique()
    
    # calculate the number of columns based on the given number of rows and unique groups
    num_cols = (len(unique_groups) + num_rows - 1) // num_rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharex=True)
    axes = axes.flatten()

    for i, group in enumerate(unique_groups):
        data_for_group = df[df[group_var] == group]
        
        sns.lineplot(data=data_for_group, x="year", y=y_var, hue='crop', ci=95, estimator=metric, ax=axes[i])
        
        axes[i].set_title(group, position=(0.5, 1.02))
        axes[i].set_xlabel('year')
        axes[i].set_ylabel(f'{metric} {y_var}')
        axes[i].grid(True)
        axes[i].legend(title='Crop')

    # remove any remaining empty subplots
    for j in range(len(unique_groups), len(axes)):
        fig.delaxes(axes[j])

    # adjust the layout to prevent overlap and set a centered supertitle
    plt.tight_layout()
    fig.suptitle(f'time-series of {metric} {y_var} for each {group_var}', y=1.05, fontsize=20)
    plt.savefig(f"{pathout_plots}{metric}_{y_var}_years_{group_var}.png", bbox_inches='tight', dpi=300)
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
        if 1 <= gap_size <= 100:
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
    
        # After the loop, let's print if there are any interpolated values
    if series.isna().sum() != interpolated_values.isna().sum():
        print("Interpolation occurred!")

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


def compute_aggregations_and_merge(df, variables_to_compute, new_variable_names, metrics, group_by_columns):
    """
    Computes the specified aggregation metric (e.g., median, sum) of given variables 
    (grouped by the provided grouping columns) and merges the results back to the original dataframe with new variable names.
    
    Parameters:
    - df: Pandas DataFrame
    - variables_to_compute: List of column names for which the metrics need to be computed
    - new_variable_names: List of new column names for the computed metrics
    - metrics: List of aggregation methods (e.g., 'median', 'sum')
    - group_by_columns: List of columns on which to group by for aggregation
    
    Returns:
    - Updated DataFrame
    """
    
    if not (len(variables_to_compute) == len(new_variable_names) == len(metrics)):
        raise ValueError("All input lists should have the same length.")
    
    for variable, new_name, metric in zip(variables_to_compute, new_variable_names, metrics):
        # Compute the specified aggregation for each variable
        aggregation = df.groupby(group_by_columns)[variable].agg(metric).astype(float).reset_index()
        aggregation = aggregation.rename(columns={variable: new_name})
        
        # Merge the computed aggregation back to the original dataframe
        df = pd.merge(df, aggregation, on=group_by_columns, how='left')
        
    # After all aggregations, replace zeros with NaNs where the original variable was NaN
    for variable, new_name in zip(variables_to_compute, new_variable_names):
        mask = df[variable].isna()
        df.loc[mask, new_name] = np.nan

    return df



















