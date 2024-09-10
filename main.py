import streamlit as st

from src import ModelTrainer

data_path = "./data/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip"
# Gender
Gender = st.selectbox("Select Gender: ", ["Female", "Male"])
# Age
Age = st.slider("Enter Age:", 0, 120, 30)
# family history ow
FHO = st.selectbox(
    "Do you have a family member who has suffered or suffers from overweight?",
    ["yes", "no"],
)
# FAVC (high kcal freq?)
FAVC = st.selectbox("Do you usually eat high caloric food?", ["yes", "no"])
# vegetables on meals
FCVC = st.slider("Do you usually eat vegetables?", 1, 3, 2)
# daily meals
NCP = st.slider("How many main meals do you have daily?", 0, 8, 5)
# foods between meals
CAEC = st.selectbox(
    "Do you eat any food between meals?", ["no", "Sometimes", "Frequently", "Always"]
)
#
SMOKE = st.selectbox("Do you usually smoke?", ["yes", "no"])
# Water
CH2O = st.slider("How much water do you drink daily?", 0.00, 4.00, 2.00)
SCC = st.selectbox("Do you monitor the calories you eat daily?", ["yes", "no"])
FAF = st.slider("How often do you have physical activity?", 0, 3, 1)
TUE = st.slider(
    "How much time do you use technological devices such as cell phone, videogames, television, computer and others?",
    0,
    2,
    0,
)
CALC = st.selectbox(
    "How often do you drink alcohol?", ["no", "Sometimes", "Frequently", "Always"]
)
MTRANS = st.selectbox(
    "Which transportation do you usually use?",
    ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
)

features = [
    Gender,
    Age,
    FHO,
    FAVC,
    FCVC,
    NCP,
    CAEC,
    SMOKE,
    CH2O,
    SCC,
    FAF,
    TUE,
    CALC,
    MTRANS,
]

if st.button("Predict"):
    st.write("Training model, please, wait!")
    pat = ModelTrainer(features)
    pat.predict_NObeyesdad(data_path)
    obe_class = pat.NObeyesdad
    st.write(f"This patient has the following weight class: {obe_class}")
