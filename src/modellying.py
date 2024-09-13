import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from .model import ModelBase
from . import DataPreprocessor
from .constants import *


class ModelTrainer(ModelBase):
    def __init__(self, features: list):
        super().__init__(features)
        self.model = None

    def train_model(self, df: pd.DataFrame):
        features_columns = df.columns[~df.columns.isin(["NObeyesdad", "Height", "Weight"])]
        X, Y = df[features_columns], df["NObeyesdad"]
        parameters = {"max_depth": [None], "n_estimators": [1000]}
        rf = RandomForestClassifier()
        clf = GridSearchCV(rf, parameters, return_train_score=True)
        clf.fit(X, Y)
        self.model = clf


    def predict_NObeyesdad(self, path: str) -> None:
        df = pd.read_csv(path)
        dp = DataPreprocessor()
        df_categ = dp.preprocess_data(df)
        self.train_model(df_categ)
        feat = dict(map(lambda i, j: (i, j), columns, self.features))
        f = pd.DataFrame(feat, index=[0])
        f[target] = None
        features_prep = dp.preprocess_input_pred(f)
        features_prep.drop(columns=[target], inplace=True)
        prediction = self.model.predict(features_prep.values)
        target_decoded = dp.inverse_encoding_target(np.array(prediction))
        self.NObeyesdad = inv_map[target_decoded[0]]
