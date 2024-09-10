from abc import ABC, abstractmethod
import pandas as pd

class DataPreprocessorBase(ABC):
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocess_input_pred(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
