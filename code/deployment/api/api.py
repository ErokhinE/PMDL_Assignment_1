from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your trained model
model = joblib.load('random_forest_model.pkl')

app = FastAPI()

class InputData(BaseModel):
    Age: float
    Gender: int
    Education: int
    Introversion_score: float
    Sensing_score: float
    Thinking_score: float
    Judjing_score: float
    Interest: int

# def preprocess(features):
#     le=LabelEncoder()
#     for feature in features:
#         if feature.dtype()=='object':
#             feature=le.fit_transform(feature)
        
#     return features


@app.post("/predict")
def predict(X: InputData):
    list_of_input = [
    X.Age,
    X.Gender,
    X.Education, 
    X.Introversion_score,
    X.Sensing_score,
    X.Thinking_score, 
    X.Judjing_score, 
    X.Interest]
    features = np.array(list_of_input).reshape(1,-1)
    print(features)
    prediction = model.predict(features)
    print(prediction)
    return {"prediction": prediction[0].item()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the model prediction API"}
