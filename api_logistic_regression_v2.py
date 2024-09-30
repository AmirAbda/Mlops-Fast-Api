import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import joblib
import pandas as pd
from typing import Optional

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables or set defaults
MODEL_PATH = os.getenv('MODEL_PATH', 'models/logistic_regression_model.joblib')

# Initialize FastAPI app
app = FastAPI(
    title="Iris Species Prediction API",
    version="1.0.0",
    description="An API to predict Iris species based on flower measurements."
)

# Load the saved pipeline
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}.")
    raise
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise

# Define the request schema with validations
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in centimeters")
    sepal_width: float = Field(..., gt=0, description="Sepal width in centimeters")
    petal_length: float = Field(..., gt=0, description="Petal length in centimeters")
    petal_width: float = Field(..., gt=0, description="Petal width in centimeters")

# Custom Exception Handler for Validation Errors
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Root Endpoint
@app.get("/", summary="Root Endpoint", description="Welcome to the Iris Species Prediction API.")
def read_root():
    return {"message": "Welcome to the Iris Species Prediction API. Use /predict to get predictions."}

# Prediction Endpoint
@app.post("/predict", summary="Predict Iris Species", description="Predict the Iris species based on flower measurements.")
async def predict_species(features: IrisFeatures):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        logger.info(f"Received input data: {input_data.to_dict(orient='records')}")

        # Make prediction
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction: {prediction}")

        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Health Check Endpoint
@app.get("/health", summary="Health Check", description="Health check to verify if the API is running.")
def health_check():
    return {"status": "API is running"}

# Example Response Model (Optional)
class PredictionResponse(BaseModel):
    prediction: str

# Update predict_species to use Response Model
@app.post("/predict", response_model=PredictionResponse, summary="Predict Iris Species", description="Predict the Iris species based on flower measurements.")
async def predict_species(features: IrisFeatures):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        logger.info(f"Received input data: {input_data.to_dict(orient='records')}")

        # Make prediction
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction: {prediction}")

        return PredictionResponse(prediction=prediction)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
