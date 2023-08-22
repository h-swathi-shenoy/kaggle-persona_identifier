import pandas as pd
import typing as t


def group_feats(col_list: list, input_df: pd.DataFrame, new_col: str) -> pd.DataFrame:
    """
    Group the feats based on the question and options

    """
    merge_df = input_df.copy(deep=True)
    false_index = []
    true_index = []
    for col in col_list:
        if "None" in col or "No" in col:
            false_list = list(set(input_df[input_df[col] == 1].index))
            false_index.append(false_list)
        else:
            true_list = list(set(input_df[input_df[col] == 1].index))
            true_index.append(true_list)
    merge_df.loc[true_list, new_col] = 1
    merge_df.loc[false_list, new_col] = 0
    merge_df = merge_df.fillna(0)
    return merge_df


def count_selected_options(
    col_list: t.List[str], input_df: pd.DataFrame, new_col: str
) -> pd.DataFrame:
    """
    Count the options for a particular question belonging to a category
    """
    input_df[new_col] = 0
    for index, row in input_df.iterrows():
        count = 0
        for col in col_list:
            if "None" in col or "No" in col:
                pass
            else:
                value = input_df.loc[index, col]
                if value:
                    count += 1
                else:
                    count = count
        input_df.loc[index, new_col] = count
    return input_df
