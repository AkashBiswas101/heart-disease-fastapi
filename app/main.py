from fastapi import FastAPI
from app.schemas import HeartFeatures

import joblib
import numpy as np

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

model = joblib.load("model/heart_model.joblib")

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "features": FEATURES
    }


@app.post("/predict")
def predict(data: HeartFeatures):

    values = np.array([
        [
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]
    ])

    prediction = model.predict(values)[0]

    return {
        "heart_disease": bool(prediction)
    }