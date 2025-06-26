import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data, apply_label
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice
)


def test_apply_label():
    """
    Test the apply_label function returns correct string labels.
    """
    # Test case 1: prediction is 1 (>50K)
    prediction_high = np.array([1])
    result_high = apply_label(prediction_high)
    assert result_high == ">50K", f"Expected '>50K', got {result_high}"
    assert isinstance(result_high, str), f"Expected string, got {type(result_high)}"
    
    # Test case 2: prediction is 0 (<=50K)
    prediction_low = np.array([0])
    result_low = apply_label(prediction_low)
    assert result_low == "<=50K", f"Expected '<=50K', got {result_low}"
    assert isinstance(result_low, str), f"Expected string, got {type(result_low)}"


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns expected types and values.
    """
    # Create simple test data where we know the expected results
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Test return types
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float" 
    assert isinstance(fbeta, float), "F-beta should be a float"
    
    # Test value ranges
    assert 0 <= precision <= 1, f"Precision {precision} should be between 0 and 1"
    assert 0 <= recall <= 1, f"Recall {recall} should be between 0 and 1"
    assert 0 <= fbeta <= 1, f"F-beta {fbeta} should be between 0 and 1"
    
    # Test with perfect predictions
    y_perfect = np.array([1, 0, 1, 0])
    y_perfect_pred = np.array([1, 0, 1, 0])
    p_perfect, r_perfect, f_perfect = compute_model_metrics(y_perfect, y_perfect_pred)
    
    assert p_perfect == 1.0, f"Perfect precision should be 1.0, got {p_perfect}"
    assert r_perfect == 1.0, f"Perfect recall should be 1.0, got {r_perfect}"
    assert f_perfect == 1.0, f"Perfect f-beta should be 1.0, got {f_perfect}"


def test_train_model_algorithm():
    """
    Test that the train_model function returns a RandomForestClassifier model.
    """
    # Create simple training data
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Test that it returns the expected algorithm
    assert isinstance(model, RandomForestClassifier), f"Expected RandomForestClassifier, got {type(model)}"
    
    # Test that the model has expected attributes
    assert hasattr(model, 'n_estimators'), "Model should have n_estimators attribute"
    assert hasattr(model, 'predict'), "Model should have predict method"
    assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
    
    # Test expected hyperparameters
    assert model.n_estimators == 100, f"Expected 100 estimators, got {model.n_estimators}"
    assert model.random_state == 42, f"Expected random_state 42, got {model.random_state}"


def test_inference_function():
    """
    Test that inference function returns predictions of expected type and shape.
    """
    # Create and train a simple model
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    model = train_model(X_train, y_train)
    
    # Test inference
    X_test = np.random.rand(20, 10)
    predictions = inference(model, X_test)
    
    # Test return type and shape
    assert isinstance(predictions, np.ndarray), f"Expected numpy array, got {type(predictions)}"
    assert predictions.shape == (20,), f"Expected shape (20,), got {predictions.shape}"
    
    # Test that predictions are binary (0 or 1)
    assert all(pred in [0, 1] for pred in predictions), "All predictions should be 0 or 1"


def test_data_loading_and_splitting():
    """
    Test that data loading and splitting produces expected sizes and types.
    """
    # Load the census data
    data = pd.read_csv("data/census.csv")
    
    # Test data properties
    assert isinstance(data, pd.DataFrame), f"Expected DataFrame, got {type(data)}"
    assert len(data) > 30000, f"Expected >30000 rows, got {len(data)}"
    assert len(data.columns) == 15, f"Expected 15 columns, got {len(data.columns)}"
    assert 'salary' in data.columns, "Data should contain 'salary' column"
    
    # Test train/test split
    train_data, test_data = train_test_split(
        data, 
        test_size=0.20, 
        random_state=42,
        stratify=data['salary']
    )
    
    # Test split sizes
    expected_train_size = int(len(data) * 0.8)
    expected_test_size = len(data) - expected_train_size
    
    assert abs(len(train_data) - expected_train_size) <= 1, f"Train size should be ~{expected_train_size}, got {len(train_data)}"
    assert abs(len(test_data) - expected_test_size) <= 1, f"Test size should be ~{expected_test_size}, got {len(test_data)}"
    
    # Test that both sets have the same columns
    assert list(train_data.columns) == list(test_data.columns), "Train and test should have same columns"


def test_process_data_types():
    """
    Test that process_data function returns expected data types and shapes.
    """
    # Create test data
    test_data = pd.DataFrame({
        'age': [25, 30, 35],
        'education': ['Bachelors', 'Masters', 'HS-grad'],
        'sex': ['Male', 'Female', 'Male'],
        'salary': ['>50K', '<=50K', '>50K']
    })
    
    categorical_features = ['education', 'sex']
    
    # Process data
    X_processed, y_processed, encoder, lb = process_data(
        test_data, 
        categorical_features=categorical_features, 
        label='salary', 
        training=True
    )
    
    # Test return types
    assert isinstance(X_processed, np.ndarray), f"X should be numpy array, got {type(X_processed)}"
    assert isinstance(y_processed, np.ndarray), f"y should be numpy array, got {type(y_processed)}"
    
    # Test shapes
    assert X_processed.shape[0] == 3, f"X should have 3 rows, got {X_processed.shape[0]}"
    assert len(y_processed) == 3, f"y should have length 3, got {len(y_processed)}"
    
    # Test that features are binary encoded
    assert X_processed.shape[1] > len(categorical_features), "One-hot encoding should increase feature count"
    
    # Test labels are binary
    assert all(label in [0, 1] for label in y_processed), "Labels should be binary (0 or 1)"


def test_model_persistence():
    """
    Test that save_model and load_model work correctly.
    """
    # Create and train a simple model
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 2, 20)
    original_model = train_model(X_train, y_train)
    
    # Test saving
    test_path = "test_model_temp.pkl"
    save_model(original_model, test_path)
    
    # Test loading
    loaded_model = load_model(test_path)
    
    # Test that loaded model is same type
    assert isinstance(loaded_model, RandomForestClassifier), f"Loaded model should be RandomForest, got {type(loaded_model)}"
    
    # Test that models make same predictions
    X_test = np.random.rand(5, 5)
    original_preds = inference(original_model, X_test)
    loaded_preds = inference(loaded_model, X_test)
    
    assert np.array_equal(original_preds, loaded_preds), "Original and loaded models should make same predictions"
    
    # Clean up
    import os
    os.remove(test_path)
