from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("diabetes_model.pkl")

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API v1 & v2 are live"}

@app.post("/v1/predict")
@app.post("/v2/predict")  # 👈 Duplicate the same logic for v2
def predict(data: DiabetesInput):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    return {"diabetic": bool(prediction)}
