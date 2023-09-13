import pandas as pd
import pytest
import os
import numpy.testing as npt
import numpy as np
from src import global_food_shocks as gfs

#--------------------------------------------------------------------------
# parameterized test for the rename_columns function
@pytest.mark.parametrize("input_df, rename_dict, expected_cols, expected_exception", [
    (pd.DataFrame({'old1': [1], 'old2': [2]}), {'old1': 'new1'}, ['new1', 'old2'], None),
    (pd.DataFrame({'old1': [1], 'old2': [2]}), {'old2': 'new2'}, ['old1', 'new2'], None),
    (pd.DataFrame({'old1': [1], 'old2': [2]}), {'old1': 'new_name', 'old2': 'new_name'}, None, ValueError),
    (pd.DataFrame({'old1': [1], 'old2': [2]}), {}, ['old1', 'old2'], None),
])
def test_rename_columns_parameterized(input_df, rename_dict, expected_cols, expected_exception):
    """
    Test the rename_columns function from global_food_shocks module.

    Scenarios being tested:
    1. Renaming one column from 'old1' to 'new1'.
    2. Renaming one column from 'old2' to 'new2'.
    3. Expecting an error when two different columns are renamed with the same name.
    4. No renaming is performed when an empty renaming dictionary is provided.

    Args:
    - input_df (pd.DataFrame): The DataFrame used as input for the rename_columns function.
    - rename_dict (dict): Dictionary with old column names as keys and new names as values.
    - expected_cols (list): Expected column names after renaming.
    - expected_exception (Exception): The type of exception expected to be raised. None if no exception is expected.

    Returns:
    - None. Function is intended for use with pytest to validate the correctness of rename_columns.
    """
    
    if expected_exception:
        with pytest.raises(expected_exception):
            gfs.rename_columns(input_df, rename_dict)
    else:
        gfs.rename_columns(input_df, rename_dict)
        assert list(input_df.columns) == expected_cols


#--------------------------------------------------------------------------
# parameterized tests for replace_values_in_columns()

@pytest.mark.parametrize(
    "data, columns_to_modify, values_to_replace, expected_result",
    [
        (
            {
                'yield_tonnes_per_ha': ['', ' ', 0, 5, 10],
                'region': ['', ' ', 0, 'North', 'South'],
                'hectares': ['', ' ', 0, 100, 200],
                'tonnes': ['', ' ', 0, 50, 100]
            },
            ['yield_tonnes_per_ha', 'region', 'hectares', 'tonnes'],
            ['', ' ', 0],
            {
                'yield_tonnes_per_ha': [np.nan, np.nan, np.nan, 5, 10],
                'region': [np.nan, np.nan, np.nan, 'North', 'South'],
                'hectares': [np.nan, np.nan, np.nan, 100, 200],
                'tonnes': [np.nan, np.nan, np.nan, 50, 100]
            }
        ),
        # Add more test cases as needed
    ]
)
def test_replace_values(data, columns_to_modify, values_to_replace, expected_result):
    """
    Test the replace_values_in_columns function with different input data.

    Parameters:
    - data (dict): Input data in the form of a dictionary representing a DataFrame.
    - columns_to_modify (list): List of columns to modify in the DataFrame.
    - values_to_replace (list): List of values to replace in the specified columns.
    - expected_result (dict): Expected result in the form of a dictionary representing the DataFrame after replacement.

    This test function verifies that the replace_values_in_columns function correctly replaces values in the DataFrame.

    Returns:
    None
    """
    df = pd.DataFrame(data)
    result_df = gfs.replace_values_in_columns(df.copy(), columns_to_modify, values_to_replace)
    expected_df = pd.DataFrame(expected_result)
    pd.testing.assert_frame_equal(result_df, expected_df)


#--------------------------------------------------------------------------
# parameterized tests for calc_miss_val_percentages()

# sample data
data = {
    'year': [2020, 2021, 2020, 2021],
    'country': ['US', 'US', 'Canada', 'Canada'],
    'crop': ['wheat', 'rice', 'wheat', 'rice'],
    'val1': [None, 10, 5, None],
    'val2': [20, None, 15, "invalid"]
}
df = pd.DataFrame(data)

# assert variable types and sizes
@pytest.mark.parametrize("df, variables_to_compute, crops, grouping_variables, exception_expected", [
    (df, ['val1', 'val2'], ['wheat', 'rice'], ['year', 'country'], False),
    ("not_a_df", ['val1'], ['wheat'], ['year'], True),
    (df, "val1", ['wheat'], ['year'], True),
    (df, [], ['wheat'], ['year'], True),
    (df, ['val1'], [], ['year'], True),
    (df, ['val1'], ['wheat'], [], True),
])
def test_input_types_and_sizes(df, variables_to_compute, crops, grouping_variables, exception_expected):
    if exception_expected:
        with pytest.raises(ValueError):
            gfs.calc_miss_val_percentages(df, variables_to_compute, crops, grouping_variables)
    else:
        result_df = gfs.calc_miss_val_percentages(df, variables_to_compute, crops, grouping_variables)


# 'missing_percentage' must be a value between 0 and 100
@pytest.mark.parametrize("variables_to_compute, crops, grouping_variables", [
    (['val1', 'val2'], ['wheat', 'rice'], ['year', 'country']),
    #... (other valid combinations)
])
def test_missing_percentage_range(variables_to_compute, crops, grouping_variables):
    result_df = gfs.calc_miss_val_percentages(df, variables_to_compute, crops, grouping_variables)
    for variable in variables_to_compute:
        assert all(0 <= value <= 100 for value in result_df[variable]), f"Invalid value in {variable}"


# 'variable' must be numerical
@pytest.mark.parametrize("variables_to_compute, crops, grouping_variables", [
    (['val1', 'val2'], ['wheat', 'rice'], ['year', 'country']),
    #... (other valid combinations)
])
def test_variable_numerical(variables_to_compute, crops, grouping_variables):
    result_df = gfs.calc_miss_val_percentages(df, variables_to_compute, crops, grouping_variables)
    for variable in variables_to_compute:
        # convert non-numeric values to NaN
        df[variable] = pd.to_numeric(df[variable], errors='coerce')
        assert all(isinstance(value, (int, float)) or pd.isna(value) for value in df[variable]), f"Non-numerical value in {variable}"



#--------------------------------------------------------------------------
# parameterized tests for compute_aggregations()

# test basic functionality
@pytest.mark.parametrize("df,variables_to_compute,new_variable_names,metrics,group_by_columns,expected_output", [
   (
        pd.DataFrame({
            'col1': [1, 2],
            'col2': [3, 4],
            'group': ['A', 'A']
        }),
        ['col1'],
        ['new_col1_sum'],
        ['sum'],
        ['group'],
        pd.DataFrame({
            'group': ['A'],
            'new_col1_sum': [3.0]
        })
    ),
])
def test_basic_functionality(df, variables_to_compute, new_variable_names, metrics, group_by_columns, expected_output):
    """
    Test basic functionality of the compute_aggregations function.

    This test checks if the function works correctly with valid inputs 
    by comparing its output to the expected DataFrame output.

    Parameters (wrapped in pytest.mark.parametrize):
        df (pd.DataFrame): Input dataframe.
        variables_to_compute (list): List of column names for metrics computation.
        new_variable_names (list): List of new column names for the computed metrics.
        metrics (list): List of aggregation methods.
        group_by_columns (list): List of columns on which to group by.
        expected_output (pd.DataFrame): Expected DataFrame output.
    """
    result = gfs.compute_aggregations(df, variables_to_compute, new_variable_names, metrics, group_by_columns)
    pd.testing.assert_frame_equal(result, expected_output, check_exact=False, atol=1e-5, rtol=1e-5)


# test list length mismatch
@pytest.mark.parametrize("variables_to_compute,new_variable_names,metrics", [
    (['col1', 'col2'], ['new_col1'], ['sum', 'mean']),
    (['col1'], ['new_col1', 'new_col2'], ['sum', 'mean']),
    (['col1', 'col2'], ['new_col1', 'new_col2'], ['sum'])
])
def test_list_length_mismatch(variables_to_compute, new_variable_names, metrics):
    """
    Test for list length mismatch.

    This test ensures the function raises a ValueError when there's a 
    mismatch in the length of `variables_to_compute`, `new_variable_names`, 
    and `metrics`.

    Parameters (wrapped in pytest.mark.parametrize):
        variables_to_compute (list): List of column names for metrics computation.
        new_variable_names (list): List of new column names for the computed metrics.
        metrics (list): List of aggregation methods.
    """
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'group': ['A', 'A', 'B']
    })
    group_by_columns = ['group']
    with pytest.raises(ValueError, match="All input lists should have the same length."):
        gfs.compute_aggregations(df, variables_to_compute, new_variable_names, metrics, group_by_columns)

# test for invalid metrics
@pytest.mark.parametrize("metrics", [
    ['invalid_metric'],
    ['mean', 'nonexistent']
])
def test_invalid_metrics(metrics):
    """
    Test for invalid metrics.

    This test checks if the function raises an exception when provided 
    with invalid metric values in the `metrics` list.

    Parameters (wrapped in pytest.mark.parametrize):
        metrics (list): List of aggregation methods.
    """
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'group': ['A', 'A', 'B']
    })
    variables_to_compute = ['col1']
    new_variable_names = ['new_col1']
    group_by_columns = ['group']
    with pytest.raises(Exception):
        gfs.compute_aggregations(df, variables_to_compute, new_variable_names, metrics, group_by_columns)

