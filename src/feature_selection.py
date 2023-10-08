import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
import numpy as np


def select_k_features(
    X_train: np.array, y_train: np.array, X_test: np.array, scoring_method="chisquare"
) -> np.array:
    """
    Perform feature selection based on the given scoring method for SelectKBest Method
    Parameters
    ----------
    X_train
    y_train
    X_test
    scoring_method

    Returns
    -------

    """
    if scoring_method == "chisquare":
        fs = SelectKBest(score_func=chi2, k="all")
    elif scoring_method == "mutualinfo":
        fs = SelectKBest(score_func=mutual_info_classif, k="all")
    else:
        raise ValueError(f"Expected chisquare or mutualinfo, given {scoring_method}")
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def get_feat_importance(
    actual_df: pd.DataFrame, X_train: np.array, y_train: np.array, X_test: np.array
) -> np.array:
    """
    Get Feature importance for the features selected by SelectKBest Method
    Parameters
    ----------
    actual_df
    X_train
    y_train
    X_test

    Returns
    -------

    """
    X_train_fs, X_test_fs, fs = select_k_features(
        X_train, y_train, X_test, scoring_method="mutualinfo"
    )
    importances = fs.scores_
    indices = np.argsort(importances)[::-1]
    train_df = actual_df.iloc[:, 1:-1].columns
    for f in range(len(train_df)):
        print(
            "%2d) %-*s %f" % (f + 1, 30, train_df[indices[f]], importances[indices[f]])
        )
    return X_train_fs, X_test_fs, indices
