import pandas as pd
import typing as t


def drop_cols(
    cols: t.List, input_df: pd.DataFrame, index_label: t.Optional[bool] = False
) -> pd.DataFrame:
    if index_label:
        input_df = input_df.drop(input_df.columns[cols], axis=1)
    else:
        input_df = input_df.drop(columns=cols)
    return input_df


def drop_cols_list(cols: t.List[list], main_df: pd.DataFrame) -> pd.DataFrame:
    for col in cols:
        main_df = main_df.drop(columns=col)
    return main_df


def gen_dict(input_df: pd.DataFrame) -> dict:
    input_shape = input_df.shape[1]
    q_count = [i for i in range(0, input_shape, 1)]
    q_dict = dict(zip(q_count, input_df.columns.tolist()))
    return q_dict


def merge_dfs(dfs: t.List[pd.DataFrame], main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concat new columns to already existing dataframe
    """
    final_df = pd.DataFrame()
    for df in dfs:
        final_df = pd.concat([final_df, df.iloc[:, -1:]], axis=1)
    main_df = pd.concat([main_df, final_df], axis=1)
    return main_df
