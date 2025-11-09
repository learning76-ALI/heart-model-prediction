from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")

# Define input schema
class InputData(BaseModel):
    age: float
    sex: int
    cp: int
    trtbps: float
    chol: float
    fbs: int
    restecg: int
    thalachh: float
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int

@app.post("/predict")
def predict(data: InputData):
    # Convert input to list of features in correct order
    features = list(data.model_dump().values())

    # Scale and predict
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)

    return {
        "prediction": prediction.tolist(),
        "input_features": data.model_dump(),
        "scaled_features": scaled_features.tolist()
    }
    