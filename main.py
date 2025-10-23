from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")

# Define input data model
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    scaled_features = scaler.transform([data.features])
    prediction = model.predict(scaled_features)
    return {"prediction": prediction.tolist(), "input_features": data.features, "scaled_features": scaled_features.tolist()}
    