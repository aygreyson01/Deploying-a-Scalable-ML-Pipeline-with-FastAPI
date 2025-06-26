#!/usr/bin/env python3
"""
Quick test to validate our model functions work
"""

import pandas as pd
import numpy as np
from ml.model import train_model, inference, save_model, load_model
from ml.data import process_data

# Create simple test data
test_data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'education': ['Bachelors', 'Masters', 'HS-grad', 'Bachelors'],
    'sex': ['Male', 'Female', 'Male', 'Female'],
    'salary': ['>50K', '<=50K', '<=50K', '>50K']
})

categorical_features = ['education', 'sex']

# Test the process_data and model functions
print("🧪 Testing model functions...")

# Process data
X_processed, y_processed, encoder, lb = process_data(
    test_data, 
    categorical_features=categorical_features, 
    label='salary', 
    training=True
)

print(f"✅ Data processed: X shape {X_processed.shape}, y shape {y_processed.shape}")

# Train model
model = train_model(X_processed, y_processed)
print("✅ Model trained successfully")

# Test inference
predictions = inference(model, X_processed)
print(f"✅ Inference works: {len(predictions)} predictions made")

# Test save/load
save_model(model, 'test_model.pkl')
print("✅ Model saved successfully")

loaded_model = load_model('test_model.pkl')
print("✅ Model loaded successfully")

# Test loaded model makes same predictions
loaded_predictions = inference(loaded_model, X_processed)
print(f"✅ Loaded model works: predictions match = {np.array_equal(predictions, loaded_predictions)}")

# Clean up
import os
os.remove('test_model.pkl')

print("\n🎉 All model functions working correctly!")
