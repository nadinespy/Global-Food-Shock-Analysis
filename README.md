# Global Food Shock Analysis

This is the repository for analyzing global yield decrease using the data from:

Anderson, W., W. Baethgen, F. Capitanio, P. Ciais, G. Cunha, L. Goddard, B. Schauberger , K. Sonder, G. Podesta, M. van der Velde, L. You, and Y. Ru. 2022. Twentieth Century Crop Statistics, 1900-2017. Palisades, New York: NASA Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/tmsp-sg82. Accessed 5th September 2023. 

Major objective is to derive yearly global yield decrease (considering wheat & maize separately) as well as food shock trends from 1900 till today, and to create in particular:
- world map with largest food shocks per country
- time-series of global yield decrease, including patterns/trends, & distribution plot for global yield decrease

## How to install

Clone the repository and create a virtual environment using poetry where you'll install all necessary dependencies:
```
$ poetry shell                                          # create virtual environment

(global-food-shock-analysis-py3.8) $ poetry install     # install necessary dependencies

(global-food-shock-analysis-py3.8) $ exit               # deactivate virtual environment
```

## How to run the analysis

Navigate to the scripts folder and execute the Jupyter notebook [analysis_global_food_shocks.ipynb`](https://github.com/nadinespy/Global-Food-Shock-Analysis/blob/main/scripts/analysis_global_food_shocks.ipynb)or alternatively execute the Python file [analysis_global_food_shocks.py](https://github.com/nadinespy/Global-Food-Shock-Analysis/blob/main/scripts/analysis_global_food_shocks.py)
Results are presented within the notebook and plots are generated in the [plots folder](https://github.com/nadinespy/Global-Food-Shock-Analysis/tree/main/results/plots).