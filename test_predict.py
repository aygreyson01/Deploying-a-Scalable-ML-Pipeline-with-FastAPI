import requests

sample_input = {
    "age": 39,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "hours_per_week": 40,
    "native_country": "United-States"
}

r = requests.post("http://127.0.0.1:8000/predict", json=sample_input)
print(f"✅ Prediction: {r.json()}")
