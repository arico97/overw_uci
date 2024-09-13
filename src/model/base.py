from abc import ABC, abstractmethod
import pandas as pd
# from sklearn.base import ClassifierMixin

class ModelBase(ABC):
    def __init__(self, features: list):
        self.features = features
        self.NObeyesdad = None

    @abstractmethod
    def train_model(self, df: pd.DataFrame) -> None:
        pass
  

    @abstractmethod
    def predict_NObeyesdad(self, path: str) -> None:
        pass
