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
    new_column_names = [rename_dict[col] if col in rename_dict else col for col in df.columns]

    if len(new_column_names) != len(set(new_column_names)):
        raise ValueError("Renaming would create duplicate column names")

    return df.rename(columns=rename_dict, inplace=True)



def replace_values_in_columns(df, columns, values_to_replace, replacement=np.nan):
    """
    Replaces specified values in given DataFrame columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame you want to modify.
    - columns (list): List of column names where the values need to be replaced.
    - values_to_replace (list): List of values you want to replace.
    - replacement (default=np.nan): The value to replace the unwanted values with.

    Returns:
    - pd.DataFrame: Modified DataFrame.
    """

    for col in columns:
        df[col].replace(values_to_replace, replacement, inplace=True)

    return df



def calc_miss_val_percentages(df, variables_to_compute, crops, grouping_variables):
    """
    Calculate missing data percentage of the specified variables.

    Parameters:
    - df: DataFrame with data
    - variables_to_compute: List of column names whose missing data percentage needs to be calculated
    - crops: List of crops for which missing data percentage is calculated
    - grouping_variables: List of columns by which data is to be grouped
    
    Returns:
    - DataFrame with columns for each variable in variables_to_compute plus the specified grouping variables
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a pandas DataFrame")
    
    if not (isinstance(variables_to_compute, list) and all(isinstance(i, str) for i in variables_to_compute) and variables_to_compute):
        raise ValueError("variables_to_compute should be a non-empty list of strings")
    
    if not (isinstance(crops, list) and all(isinstance(i, str) for i in crops) and crops):
        raise ValueError("crops should be a non-empty list of strings")
    
    if not (isinstance(grouping_variables, list) and all(isinstance(i, str) for i in grouping_variables) and grouping_variables):
        raise ValueError("grouping_variables should be a non-empty list of strings")

    # filter the dataframe for only the crops in question
    df = df[df['crop'].isin(crops)]

    group_cols = grouping_variables + ['crop']

    # create an empty dataframe to hold the results
    result_df = df.groupby(group_cols).size().reset_index().drop(0, axis=1)

    for variable in variables_to_compute:
        missing_percentage = df.groupby(group_cols).apply(lambda group: group[variable].isna().mean() * 100).reset_index(name=variable)
        missing_percentage = df.groupby(group_cols).apply(lambda group: group[variable].isna().mean() * 100).reset_index()
        missing_percentage.columns = group_cols + [variable]

        result_df = pd.merge(result_df, missing_percentage, on=group_cols, how='left')

    return result_df


def plot_miss_val_percentages(df, variables_to_compute, crops, grouping_variables, pathout):
    """
    Plot missing data percentages for all specified variables from the given DataFrame.

    Parameters:
    - df: DataFrame with columns specified in [grouping_variables], 'crop', and a list of variables 
      to compute in [variables_to_compute].
    - variables_to_compute: List of column names whose missing data percentage needs to be plotted.
    - crops: List of crops for which the missing data percentages are to be plotted.
    - grouping_variables: List of columns by which data is grouped.
    - pathout: Output path for saving the plots (without file extension).
    """
    # filter the dataframe for only the crops in question
    df = df[df['crop'].isin(crops)]

    # create a composite label for the x-axis if multiple grouping variables
    if len(grouping_variables) > 1:
        df['composite_label'] = df[grouping_variables].astype(str).agg(' | '.join, axis=1)
        x_label = 'composite_label'
    else:
        x_label = grouping_variables[0]
    
    for variable in variables_to_compute:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=x_label, y=variable, hue='crop', data=df, ci=None)
        plt.title(f"missing data percentage for {variable} grouped by {', '.join(grouping_variables)}")
        plt.ylabel("missing percentage (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(pathout + f'distr_{variable}_by_{"_".join(grouping_variables)}.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    # drop the composite label column if it was created
    if 'composite_label' in df.columns:
        df.drop('composite_label', axis=1, inplace=True)



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
    plt.suptitle(f'distributions of {plot_var} by {grouping_var}', y=1.02, fontsize=20)
    plt.savefig(pathout_plots+f'distr_{plot_var}_by_{grouping_var}.png', bbox_inches='tight', dpi=300)

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

    # filter dataframe by the crops provided
    df = df[df['crop'].isin(crops)]
    
    # if no grouping is required
    if group_var is None:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="year", y=y_var, hue='crop', ci=95, estimator=metric)
        plt.title(f'time series of {metric} of {y_var}', fontsize=20)
        plt.xlabel('year')
        plt.ylabel(f'{metric} of {y_var}')
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title='crop')
        plt.savefig(f"{pathout_plots}{metric}_{y_var}_by_{group_var}.png", bbox_inches='tight', dpi=300)
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
        axes[i].legend(title='crop')

    # remove any remaining empty subplots
    for j in range(len(unique_groups), len(axes)):
        fig.delaxes(axes[j])

    # adjust the layout to prevent overlap and set a centered supertitle
    plt.tight_layout()
    fig.suptitle(f'time series of {metric} of {y_var} for each {group_var}', y=1.05, fontsize=20)
    plt.savefig(f"{pathout_plots}{metric}_{y_var}{group_var}.png", bbox_inches='tight', dpi=300)
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
    # helper function to handle NaN sequences
    def handle_sequence(start, end, series, interpolated_values):
        gap_size = end - start
        if 1 <= gap_size <= 100:
            # ensure we're not on the edge of the series
            if start > 0 and end < len(series):
                linspace_values = np.linspace(series.iloc[start-1], series.iloc[end], gap_size+2)[1:-1]
                interpolated_values.iloc[start:end] = linspace_values

    # placeholder for interpolated values
    interpolated_values = series.copy()

    # manually loop to determine NaN sequences
    start = None
    for i in range(len(series)):
        if np.isnan(series.iloc[i]) and start is None:
            start = i
        elif start is not None and (not np.isnan(series.iloc[i]) or i == len(series)-1):
            handle_sequence(start, i, series, interpolated_values)
            start = None
    
        # after the loop, let's print if there are any interpolated values
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



def compute_aggregations(df, variables_to_compute, new_variable_names, metrics, group_by_columns):
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
    
    # compute the specified aggregation for each variable
    aggregation = df.groupby(group_by_columns).agg(
        {variable: metric for variable, metric in zip(variables_to_compute, metrics)}).astype(float).reset_index()
    aggregation = aggregation.rename(columns={variable: new_name for variable, new_name in zip(variables_to_compute, new_variable_names)})

    return aggregation




















