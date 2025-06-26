import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.model import train_model, save_model
from ml.data import process_data

# Load cleaned data
data = pd.read_csv("data/census_clean.csv")

# Process the data
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Save model and encoders
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
save_model(lb, "model/lb.pkl")
