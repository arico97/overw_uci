
mapping = {
    "Normal_Weight": "NW",
    "Overweight_Level_I": "OW1",
    "Overweight_Level_II": "OW2",
    "Obesity_Type_I": "O1",
    "Insufficient_Weight": "IW",
    "Obesity_Type_II": "O2",
    "Obesity_Type_III": "O3",
}

columns = [
    "Gender",
    "Age",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

target = 'NObeyesdad'

inv_map = {v: k for k, v in mapping.items()}

model_path = './src/trained_models/model_trained.joblib'
