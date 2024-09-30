from fastapi import FastAPI
import joblib 
from pydantic import BaseModel 
import pandas as pd 

# Load the saved pipline 
model = joblib.load('models/logistic_regression_model.joblib')

# Define the request schema 
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float 

# Initialize the FastAPI app
app = FastAPI()

def predict_sepcies(features: IrisFeatures):

    input_data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
    
    prediction = model.predict(input_data)[0]

    return {"prediction": prediction}