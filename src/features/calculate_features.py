import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from ipywidgets import IntProgress
from IPython.display import display
from tqdm import tqdm
tqdm.pandas()
import os
import sys

# project_path = os.path.abspath(os.path.join('..'))

# if project_path not in sys.path:
#     sys.path.append(project_path)

def percentile(n):
    """
    Calculate n - percentile of data
    """
    def percentile_(x):
        return np.nanpercentile(x, n)

    percentile_.__name__ = 'perc%s' % n
    return percentile_

def fill_missing_dates(x, date_col, l):
    """
    tool to create rolling lagged features
    """
    x = x.set_index(date_col)
    index = (x.index - pd.Timedelta(f'{l}W'))
    index = index.union(x.index)
    results = x.reindex(index, fill_value=np.nan).sort_index()

    return results


def calc_preag_fill(data, group_col, date_col, target_cols, preagg_method, l):
#     return data_preag
    ## fill missing dates
    data_preag_filled = data.groupby(group_col[:-1]).progress_apply(
        fill_missing_dates, date_col=date_col, l=l).drop(group_col[:-1],
                                                    axis=1).reset_index()

    ## return DataFrame with calculated preaggregation and filled missing dates
    return data_preag_filled


def calc_rolling(data_preag_filled_shifted, group_col, date_col, method, w):

    ## calc rolling stats
    lf_df = data_preag_filled_shifted.groupby(group_col[:-1]).\
        apply(lambda x: x.set_index(date_col).rolling(window=w, min_periods=1).agg(method)).drop(group_col[:-1], axis=1).reset_index()

    ## return DataFrame with rolled columns from target_vars
    return lf_df


def calc_ewm(data_preag_filled, group_col, date_col, span):
    ## calc ewm stats
    lf_df_filled = data_preag_filled.groupby(group_col[:-1]).\
        apply(lambda x: x.set_index(date_col).ewm(span=span).mean()).drop(group_col[:-1], axis=1)

    ## return DataFrame with rolled columns from target_vars
    return lf_df_filled


def shift(data_preag_filled, group_col, date_col, lag):

    data_preag_filled_shifted = data_preag_filled.set_index(date_col).groupby(
        group_col[:-1]).progress_apply(lambda x: x.shift(lag * 7, freq='D')).\
            droplevel(group_col[:-1]).reset_index()

    ## return DataFrame with following columns: filter_col, id_cols, date_col and shifted stats
    return data_preag_filled_shifted


def generate_lagged_features(
        data: pd.DataFrame,
        target_cols: list = ['Demand'],
        id_cols: list = ['SKU_id', 'Store_id'],
        date_col: str = 'Date',
        lags: list = [7, 14, 21, 28],
        windows: list = ['7D', '14D', '28D', '56D'],
        preagg_methods: list = ['mean'],
        agg_methods: list = ['mean', 'median', percentile(10), pd.Series.skew],
        dynamic_filters: list = ['weekday', 'Promo'],
        ewm_params: dict = {'weekday': [14, 28], 'Promo': [14, 42]}) -> pd.DataFrame:
    
    ''' 
    data - dataframe with default index
    target_cols - column names for lags calculation
    id_cols - key columns to identify unique values
    date_col - column with datetime format values
    lags - lag values(days)
    windows - windows(days/weeks/months/etc.),
        calculation is performed within time range length of window
    preagg_methods - applied methods before rolling to make
        every value unique for given id_cols
    agg_methods - method of aggregation('mean', 'median', percentile, etc.)
    dynamic_filters - column names to use as filter
    ewm_params - span values(days) for each dynamic_filter
    '''

    data = data.sort_values(date_col)
    out_df = deepcopy(data)
    dates = [min(data[date_col]), max(data[date_col])]

    total = len(target_cols) * len(lags) * len(windows) * len(preagg_methods) * len(agg_methods) * len(dynamic_filters)
    progress = IntProgress(min=0, max=total)
    display(progress)

    for filter_col in tqdm(dynamic_filters, leave=False):
        group_col = [filter_col] + id_cols + [date_col]
        for lag in tqdm(lags, leave=False):
            for preagg in tqdm(preagg_methods, leave=False):
                data_preag_filled = calc_preag_fill(data, group_col, date_col,
                                                    target_cols, preagg, lag)
                
                data_preag_filled_shifted = shift(data_preag_filled, group_col, date_col, lag)
                ## add ewm features
                for alpha in ewm_params.get(filter_col, []):
                    ewm = calc_ewm(data_preag_filled_shifted, group_col,
                                          date_col, alpha)
                    new_names = {x: "{0}_lag{1}_alpha{2}_key{3}_preag{4}_{5}_dynamic_ewm".\
                        format(x, lag, alpha, '_'.join(id_cols), preagg, filter_col) for x in target_cols}

                    out_df = pd.merge(out_df,
                                      ewm.rename(columns=new_names),
                                      how='left',
                                      on=group_col)

                for w in tqdm(windows, leave=False):
                    for method in tqdm(agg_methods, leave=False):
                        rolling = calc_rolling(data_preag_filled_shifted,
                                                      group_col, date_col,
                                                      method, w)

                        ## lf_df - DataFrame with following columns: filter_col, id_cols, date_col, shifted rolling stats

                        method_name = method.__name__ if type(
                            method) != str else method

                        new_names = {x: "{0}_lag{1}_w{2}_key{3}_preag{4}_ag{5}_{6}_dynamic_rolling".\
                                     format(x, lag, w, '_'.join(id_cols), preagg, method_name, filter_col) for x in target_cols}

                        out_df = pd.merge(out_df,
                                          rolling[group_col + target_cols].rename(columns=new_names),
                                          how='left',
                                          on=group_col)
                        progress.value += 1

    return out_df



if __name__ == '__main__':
    stores = [533, 535, 540, 555, 557, 562, 637, 644]

    for store in tqdm(stores):
        data = pd.read_csv(f'DataShopsNoLagsAll/{store}.csv')
        data['period_start_dt'] = pd.to_datetime(data['period_start_dt'])

        # ## tmp
        # data_sample = data.iloc[:1000]

        target_cols = ['demand']
        id_cols = ['product_rk', 'store_location_rk']
        date_col = 'period_start_dt'

        data_lagged_features = generate_lagged_features(data 
                            , target_cols = target_cols
                            , id_cols = id_cols
                            , date_col = date_col
                            , lags = [4, 8, 26, 52] ## weeks
                            , windows = ['28D', '56D', '182D', '364D'] ## days
                            , preagg_methods = ['mean'] # ['mean', 'count']
                            , agg_methods = ['mean', 'median']
                            , dynamic_filters = ['PROMO1_FLAG', 'PROMO2_FLAG', 'PROMO12_FLAG', 'NO_FILTER']
                            , ewm_params={}
                            )

        data_lagged_features.to_csv(f'{store}_features.csv', index=False)


