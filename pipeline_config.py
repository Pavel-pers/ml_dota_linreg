from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import category_encoders as ce
import sklearn.base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from shared import HeroesEncoder
from cuml.linear_model import LogisticRegression as cuLogisticRegression


@dataclass
class FeatureGroup:
    name: str
    columns: List[str]
    scaling: bool = False
    categorical: bool = False
    enabled: bool = True


@dataclass
class TextFeatureConfig:
    use_radiant_chat: bool = True
    use_dire_chat: bool = True
    vectorizer_template: Optional[TfidfVectorizer] = None
    enabled: bool = True

    def __post_init__(self):
        if self.vectorizer_template is None:
            self.vectorizer_template = TfidfVectorizer()


@dataclass
class HeroFeatureConfig:
    radiant_encoder: Any = None
    dire_encoder: Any = None
    enabled: bool = True


class LearnConfig:
    """Builder для LearnPipeline. Все мутирующие методы возвращают self."""

    def __init__(self, name: str = "default"):
        self.nasme = name
        self._groups: Dict[str, FeatureGroup] = {}
        self._group_order: List[str] = []
        self._hero_config: Optional[HeroFeatureConfig] = None
        self._custom_sparse: Dict[str, Any] = {}
        self._text_config: Optional[TextFeatureConfig] = None
        self._model_cls: Optional[Type] = None
        self._model_params: Dict[str, Any] = {}
        self._scaler_cls: Optional[Type] = StandardScaler
        self._scaler_params: Dict[str, Any] = {}
        self._encoder_cls: Type = ce.OneHotEncoder
        self._encoder_params: Dict[str, Any] = {"use_cat_names": True}
        self._use_gpu: bool = False

    # --- Dense features

    def add_group(self, name: str, columns: List[str],
                  scaling: bool = False, categorical: bool = False):
        self._groups[name] = FeatureGroup(name, list(columns), scaling, categorical)
        self._group_order.append(name)

    def get_group(self, name: str) -> Optional[FeatureGroup]:
        return self._groups.get(name)

    def toggle_group(self, name: str, enabled: bool = True):
        if name in self._groups:
            self._groups[name].enabled = enabled

    # --- Hero features

    def add_hero_feature(self, radiant_encoder: Any = None,
                         dire_encoder: Any = None):
        if radiant_encoder is None or dire_encoder is None:
            from shared import HeroesEncoder
            radiant_encoder = radiant_encoder or HeroesEncoder(pop_value=1)
            dire_encoder = dire_encoder or HeroesEncoder(pop_value=-1)
        self._hero_config = HeroFeatureConfig(deepcopy(radiant_encoder), deepcopy(dire_encoder))

    def toggle_hero_feature(self, enabled: bool = True):
        if self._hero_config:
            self._hero_config.enabled = enabled

    # --- Тext features

    def add_text_feature(self, vectorizer_template: Optional[TfidfVectorizer] = None,
                         use_radiant_chat: bool = True,
                         use_dire_chat: bool = True):
        self._text_config = TextFeatureConfig(
            use_radiant_chat, use_dire_chat, deepcopy(vectorizer_template or TfidfVectorizer()))

    def toggle_text_feature(self, enabled: bool = True):
        if self._text_config:
            self._text_config.enabled = enabled

    # --- Sparce encoders

    def add_sparse_feature(self, column_name: str, encoder: Any):
        self._custom_sparse[column_name] = deepcopy(encoder)

    # --- Linear model

    def set_model(self, model_cls: Type, **params: Any):
        self._model_cls = model_cls
        self._model_params = dict(params)

    # --- Scaler

    def set_scaler(self, scaler_cls: Optional[Type] = StandardScaler, **params: Any):
        self._scaler_cls = scaler_cls
        self._scaler_params = dict(params)

    # --- encoder

    def set_encoder(self, encoder_cls: Type = ce.OneHotEncoder, **params: Any):
        self._encoder_cls = encoder_cls
        self._encoder_params = dict(params)

    # --- Props

    @property
    def active_groups(self) -> List[FeatureGroup]:
        return [self._groups[gr] for gr in self._group_order
                if gr in self._groups and self._groups[gr].enabled]

    @property
    def all_groups(self) -> List[FeatureGroup]:
        return [self._groups[gr] for gr in self._group_order if gr in self._groups]

    @property
    def dense_columns(self) -> List[str]:
        return [col for gr in self.active_groups for col in gr.columns]

    @property
    def scaling_columns(self) -> List[str]:
        return [col for gr in self.active_groups if gr.scaling for col in gr.columns]

    @property
    def categorical_columns(self) -> List[str]:
        return [col for gr in self.active_groups if gr.categorical for col in gr.columns]

    @property
    def has_heroes(self) -> bool:
        return self._hero_config is not None and self._hero_config.enabled

    @property
    def has_text(self) -> bool:
        return self._text_config is not None and self._text_config.enabled

    # --- Build

    def pipeline_args(self) -> Dict[str, Any]:
        cat_cols = self.categorical_columns
        encoder = self._encoder_cls(cols=cat_cols, **self._encoder_params) if cat_cols else None

        scaler = None
        if self.scaling_columns and self._scaler_cls is not None:
            scaler = self._scaler_cls(**self._scaler_params)

        sparse_features: Dict[str, Any] = {}
        if self.has_heroes:
            hc = self._hero_config
            sparse_features["heroes_radiant"] = deepcopy(hc.radiant_encoder)
            sparse_features["heroes_dire"] = deepcopy(hc.dire_encoder)
        for col, enc in self._custom_sparse.items():
            sparse_features[col] = deepcopy(enc)

        text_features: Dict[str, TfidfVectorizer] = {}
        if self.has_text:
            tc = self._text_config
            if tc.use_radiant_chat:
                text_features["radiant_chat"] = deepcopy(tc.vectorizer_template)
            if tc.use_dire_chat:
                text_features["dire_chat"] = deepcopy(tc.vectorizer_template)

        return dict(
            dense_features=self.dense_columns,
            scaling_features=self.scaling_columns,
            encoder=encoder,
            scaler=scaler,
            model=self._model_cls(**self._model_params),
            text_features=text_features,
            sparse_features=sparse_features,
        )

    # --- Clone ---

    def clone(self, new_name: Optional[str] = None):
        cloned = deepcopy(self)
        if new_name is not None:
            cloned.name = new_name
        return cloned

    # --- Optuna helpers ---

    def suggest_group_toggles(self, trial: Any,
                              group_names: Optional[List[str]] = None):
        for name in (group_names or list(self._groups.keys())):
            if name in self._groups:
                self.toggle_group(name, trial.suggest_categorical(f"use_{name}", [True, False]))


    def suggest_hero_encoder(self, trial: Any, *,
                             allow_disable: bool = False):
        if allow_disable:
            use = trial.suggest_categorical("use_heroes", [True, False])
            if not use:
                self.toggle_hero_feature(False)
                return None

        self._hero_config = HeroFeatureConfig(
            HeroesEncoder(pop_value=1), HeroesEncoder(pop_value=-1))
        return None

    def suggest_text_feature(self, trial: Any, *,
                             max_features_range: tuple = (250, 5000),
                             min_df_range: tuple = (5, 100),
                             max_df_range: tuple = (0.8, 0.99),
                             allow_disable: bool = True):
        use_r = trial.suggest_categorical("use_radiant_chat", [True, False])
        use_d = trial.suggest_categorical("use_dire_chat", [True, False])

        if allow_disable and not use_r and not use_d:
            self.toggle_text_feature(False)
            return None

        if not use_r and not use_d:
            use_r = True

        max_features = trial.suggest_int("text_max_features", *max_features_range, step=250)
        min_df = trial.suggest_int("text_min_df", *min_df_range)
        max_df = trial.suggest_float("text_max_df", *max_df_range)
        sublinear_tf = trial.suggest_categorical("text_sublinear_tf", [True, False])
        ngram_max = trial.suggest_int("text_ngram_max", 1, 2)

        vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df,
            sublinear_tf=sublinear_tf, ngram_range=(1, ngram_max))

        self.add_text_feature(vectorizer, use_radiant_chat=use_r, use_dire_chat=use_d)
        return None

    def suggest_model(self, trial: Any, *,
                      check_sgd: bool = False,
                      C_range: tuple = (1e-4, 10.0),
                      alpha_range: tuple = (1e-6, 1e-1),
                      max_iter_range: tuple = (100, 2000),
                      solvers: List[str] = None,
                      sgd_losses: List[str] = None):
        if check_sgd and self._use_gpu:
            raise RuntimeError('cuml have no sgd')

        if solvers is None:
            solvers = ["lbfgs", "liblinear", "saga"]
        if sgd_losses is None:
            sgd_losses = ["log_loss", "hinge"]

        if check_sgd:
            model_type = trial.suggest_categorical("model_type", ["logreg", "sgd"])
        else:
            model_type = "logreg"

        max_iter = trial.suggest_int("max_iter", *max_iter_range)

        if model_type == "logreg":
            C = trial.suggest_float("C", *C_range, log=True)
            max_iter = trial.suggest_int("max_iter", *max_iter_range)
            if self._use_gpu:
                self.set_model(cuLogisticRegression, C=C, max_iter=max_iter)
            else:
                solver = trial.suggest_categorical("solver", solvers)
                self.set_model(LogisticRegression, C=C, max_iter=max_iter, solver=solver)
        else:
            alpha = trial.suggest_float("alpha", *alpha_range, log=True)
            loss = trial.suggest_categorical("sgd_loss", sgd_losses)
            self.set_model(SGDClassifier, alpha=alpha, max_iter=max_iter, loss=loss,
                           random_state=420)