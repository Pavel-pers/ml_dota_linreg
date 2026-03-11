from typing import List, Optional, Dict

import category_encoders as ce
import polars
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers import gini


class LearnPipeline:
    def __init__(self,
                 dense_features: List,
                 scaling_features: List,
                 encoder: Optional[ce.OneHotEncoder],
                 scaler: Optional[sklearn.preprocessing.StandardScaler],
                 model: sklearn.base.ClassifierMixin,
                 text_features: Dict[str, TfidfVectorizer] = None,
                 sparse_features: Dict[str, object] = None,
                 player_encoder = None,                      
                 ):

        self.dense_features = dense_features
        self.scaling_features = scaling_features
        self.encoder = encoder
        self.scaler = scaler
        self.model = model
        self.text_features = text_features or {}
        self.sparse_features = sparse_features or {}
        self.player_encoder = player_encoder                 

    def _enrich_df(self, df: polars.DataFrame, fit: bool) -> polars.DataFrame:
        if self.player_encoder is not None:
            if fit:
                df = self.player_encoder.fit_transform(df)
            else:
                df = self.player_encoder.transform(df)
        return df

    def prepare_data(self, df: polars.DataFrame, sparse_list=None, fit=False)->csr_matrix:
        if self.player_encoder is not None:
            if fit:
                df = self.player_encoder.fit_transform(df)
            else:
                df = self.player_encoder.transform(df)

        dense_feaures = df[self.dense_features].to_pandas()
        if self.encoder:
            dense_feaures = self.encoder.fit_transform(dense_feaures) if fit else self.encoder.transform(dense_feaures)
        if self.scaler:
            dense_feaures[self.scaling_features] = self.scaler.fit_transform(
                dense_feaures[self.scaling_features]) if fit else self.scaler.transform(
                dense_feaures[self.scaling_features])

        for col in dense_feaures.columns:
            if dense_feaures[col].dtype == bool or dense_feaures[col].dtype == object:
                dense_feaures[col] = dense_feaures[col].astype(float)
        result_matrix = csr_matrix(dense_feaures)

        if sparse_list:
            result_matrix = hstack([result_matrix] + sparse_list)

        for col_name, sparse_encoder in self.sparse_features.items():
            column = df[col_name]
            if fit:
                sparse_matrix = sparse_encoder.fit_transform(column)
            else:
                sparse_matrix = sparse_encoder.transform(column)
            result_matrix = hstack([result_matrix, sparse_matrix])

        for col_name, vectorizer in self.text_features.items():
            texts = df[col_name].fill_null("").to_list()
            text_matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
            result_matrix = hstack([result_matrix, text_matrix])

        return result_matrix

    def fit(self, df: polars.DataFrame, sparse_list=None, verbose=False):
        features = self.prepare_data(df, sparse_list, fit=True)
        if verbose:
            print(f'fit on {features.shape} shape, df columns:', df.columns)
        target = df['radiant_win'].to_numpy()
        self.model.fit(features, target)

    def predict(self, X, sparse_list=None, verbose=False):
        X_processed = self.prepare_data(X, sparse_list)
        if verbose:
            print(f'predict on {X_processed.shape} shape, df columns:', X.columns)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_processed)[:, 1]
        return self.model.decision_function(X_processed)

    def evaluate(self, df: polars.DataFrame, sparse_list: List[csr_matrix], split_masks: polars.DataFrame, verbose=False) -> List[float]:
        scores = []
        for idx_train, idx_val in split_masks:
            train_df = df[idx_train]
            validate_df = df[idx_val]
            if sparse_list is not None:
                train_sparse = [s[idx_train] for s in sparse_list]
                validate_sparse = [s[idx_val] for s in sparse_list]
            else:
                train_sparse = None
                validate_sparse = None

            self.fit(train_df, train_sparse, verbose=verbose)
            X_validate_df = validate_df.select(polars.col("*").exclude("radiant_win"))
            y_validate = validate_df['radiant_win']
            predict = self.predict(X_validate_df, validate_sparse, verbose=verbose)
            scores.append(gini(y_validate.to_numpy(), predict))

        return scores

    def get_features_names(self):
        names = []
        if self.encoder:
            names += list(self.encoder.get_feature_names_out())
        else:
            names += self.dense_features

        for col_name, sparse_enc in self.sparse_features.items():
            names += [f"{col_name}_{key}" for key in sparse_enc.get_keys()]

        for col_name, vectorizer in self.text_features.items():
            names += [f"{col_name}_{word}" for word in vectorizer.get_feature_names_out()]

        return names

    def get_weights(self):
        feature_names = self.get_features_names()
        weights = self.model.coef_[0]
        weight_df = polars.DataFrame(
            {
                "feature": feature_names,
                "weight": weights
            }
        )
        return weight_df
    