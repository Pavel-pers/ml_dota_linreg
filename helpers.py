import numpy as np
import polars
import sklearn.model_selection
from sklearn.metrics import roc_auc_score


def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1.0


def get_cv_oot_split(cur_df: polars.DataFrame, n_splits: int = 11):
    train_cv_oot = sklearn.model_selection.TimeSeriesSplit(n_splits=n_splits)

    for train_index, validate_index in train_cv_oot.split(cur_df):
        train, validate = cur_df[train_index], cur_df[validate_index]
        yield train, validate


def get_oot_split(cur_df: polars.DataFrame, trashold: str = "2024-11-01"):
    train_oot = cur_df.filter(polars.col('date').cast(polars.String) < trashold)
    validate_oot = cur_df.filter(polars.col('date').cast(polars.String) >= trashold)

    return [(train_oot, validate_oot)]

def get_oot_split_mask(cur_df: polars.DataFrame, trashold: str = "2024-11-01"):
    mask_train = (cur_df['date'].cast(polars.String) < trashold).to_numpy().astype(bool)
    mask_val = ~mask_train

    idx_train = np.where(mask_train)[0]
    idx_val = np.where(mask_val)[0]

    return [(idx_train, idx_val)]
