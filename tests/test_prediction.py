import pytest
import pandas as pd

from src import DataPreprocessor
from src import ModelTrainer

# Path to the CSV file
CSV_FILE_PATH = 'data/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip'

def test_preprocess_data():
    dp = DataPreprocessor()
    df = pd.read_csv(CSV_FILE_PATH)
    processed_df = dp.preprocess_data(df)
    assert 'NObeyesdad' in processed_df.columns
    assert processed_df['NObeyesdad'].dtype == 'int64'

def test_preprocess_input_pred():
    dp = DataPreprocessor()
    input_df = pd.read_csv(CSV_FILE_PATH)
    input_df = input_df.head(1)  # Use only the first row for prediction
    preprocessed_df = dp.preprocess_input_pred(input_df)
    assert preprocessed_df.shape == input_df.shape

def test_train_model():
    model_trainer = ModelTrainer()
    df = pd.read_csv(CSV_FILE_PATH)
    model = model_trainer.train_model(df)
    assert model is not None
    assert hasattr(model_trainer, 'model')

def test_predict():
    model_trainer = ModelTrainer()
    df = pd.read_csv(CSV_FILE_PATH)
    model_trainer.train_model(df)
    
    features = ["Male", 30, 1, 1, 1, 2, 1, 0, 1, 0, 3, 0, 1, 2]
    f = pd.DataFrame(dict(zip([
        "Gender", "Age", "family_history_with_overweight", "FAVC", "FCVC",
        "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
    ], features)), index=[0])
    
    prediction = model_trainer.predict(f)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2, 3, 4, 5, 6]  # Ensure predictions are in expected range

def test_predict_NObeyesdad():
    features = ["Male", 30, 1, 1, 1, 2, 1, 0, 1, 0, 3, 0, 1, 2]
    predictor = ModelTrainer(features)
    
    # Use the same CSV file for prediction
    predictor.predict_NObeyesdad(CSV_FILE_PATH)
    assert predictor.NObeyesdad in ["Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Insufficient_Weight", "Obesity_Type_II", "Obesity_Type_III"]

if __name__ == "__main__":
    pytest.main()
