from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CensusInput(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int
    native_country: str

@app.get("/")
def root():
    return {"message": "Hello from the Census Income Prediction API!"}

@app.post("/predict")
def predict(data: CensusInput):
    # Dummy logic
    if data.hours_per_week > 30:
        return {"prediction": ">=50K"}
    else:
        return {"prediction": "<=50K"}
