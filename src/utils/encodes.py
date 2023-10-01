import pandas as pd


def label_encode_bind(df: pd.DataFrame, col: str) -> pd.DataFrame:
    col_count = df[col].value_counts()
    value_dict = pd.Series(range(len(col_count)), index=col_count.index)
    df = df[col].map(value_dict)
    return df


def gen_encode_cols(input_df: pd.DataFrame) -> pd.DataFrame:
    encode_df = pd.DataFrame()
    for col in input_df.columns:
        sub_df = input_df[[col]]
        map_df = label_encode_bind(sub_df, col)
        encode_df = pd.concat([encode_df, map_df], axis=1)
    return encode_df
