import pandas as pd
import numpy as np

from .data_preprocessing import DataPreprocessorBase
from .constants import *

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

global oe 
global le

oe = OrdinalEncoder()
le = LabelEncoder()

class DataPreprocessor(DataPreprocessorBase):
    def encode_feature(self, df: pd.DataFrame) -> np.array:
       return oe.fit_transform(df.values)
    
    def only_encode_features(self, df: pd.DataFrame) -> np.array:
        return oe.transform(df.values)
    
    def inverse_encoding_target(self, prediction: np.array) -> np.array:
        return le.inverse_transform(prediction)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.NCP = df.NCP.round()
        df.FAF = df.FAF.round()
        df.TUE = df.TUE.round()
        df.rename(columns={"family_history_with_overweight": "FHO"}, inplace=True)
        df.replace({"NObeyesdad": mapping}, inplace=True)
        df_ct = df[["Gender", "SMOKE", "FHO", "FAVC", "CAEC", "SCC", "CALC", "MTRANS"]]
        Y = df["NObeyesdad"]
        df_categorized = df.copy()
        df_categorized[["Gender", "SMOKE", "FHO", "FAVC", "CAEC", "SCC", "CALC", "MTRANS"]] = self.encode_feature(df_ct)
        df_categorized["NObeyesdad"] = le.fit_transform(Y)
        return df_categorized

    def preprocess_input_pred(self, df: pd.DataFrame) -> pd.DataFrame:
        df.NCP = df.NCP.round()
        df.FAF = df.FAF.round()
        df.TUE = df.TUE.round()
        df.rename(columns={"family_history_with_overweight": "FHO"}, inplace=True)
        df_ct = df[["Gender", "SMOKE", "FHO", "FAVC", "CAEC", "SCC", "CALC", "MTRANS"]]
        df_categorized = df.copy()
        df_transf = self.only_encode_features(df_ct)
        df_categorized[["Gender", "SMOKE", "FHO", "FAVC", "CAEC", "SCC", "CALC", "MTRANS"]] = df_transf
        return df_categorized
    # create decorators to pass to encode to uncode, to pandas, to array, etc. 
    
