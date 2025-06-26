import pytest
import numpy as np
import pandas as pd
from ml.data import process_data, apply_label
from ml.model import compute_model_metrics


def test_apply_label():
    """
    Test the apply_label function with both possible binary outcomes.
    """
    # Test case 1: prediction is 1 (>50K)
    prediction_high = np.array([1])
    result_high = apply_label(prediction_high)
    assert result_high == ">50K", f"Expected '>50K', got {result_high}"
    
    # Test case 2: prediction is 0 (<=50K)
    prediction_low = np.array([0])
    result_low = apply_label(prediction_low)
    assert result_low == "<=50K", f"Expected '<=50K', got {result_low}"


def test_compute_model_metrics():
    """
    Test the compute_model_metrics function with known values.
    """
    # Create simple test data where we know the expected results
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Verify all metrics are returned as floats
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float" 
    assert isinstance(fbeta, float), "F-beta should be a float"
    
    # Verify metrics are in valid range [0, 1]
    assert 0 <= precision <= 1, f"Precision {precision} should be between 0 and 1"
    assert 0 <= recall <= 1, f"Recall {recall} should be between 0 and 1"
    assert 0 <= fbeta <= 1, f"F-beta {fbeta} should be between 0 and 1"
    
    # For perfect predictions, all metrics should be 1.0
    y_perfect = np.array([1, 0, 1, 0])
    y_perfect_pred = np.array([1, 0, 1, 0])
    p_perfect, r_perfect, f_perfect = compute_model_metrics(y_perfect, y_perfect_pred)
    
    assert p_perfect == 1.0, f"Perfect precision should be 1.0, got {p_perfect}"
    assert r_perfect == 1.0, f"Perfect recall should be 1.0, got {r_perfect}"
    assert f_perfect == 1.0, f"Perfect f-beta should be 1.0, got {f_perfect}"


def test_process_data():
    """
    Test the process_data function with a simple dataset.
    """
    # Create a simple test DataFrame
    test_data = pd.DataFrame({
        'age': [25, 30, 35],
        'education': ['Bachelors', 'Masters', 'High-school'],
        'salary': ['>50K', '<=50K', '>50K']
    })
    
    categorical_features = ['education']
    
    # Test training mode
    X_processed, y_processed, encoder, lb = process_data(
        test_data, 
        categorical_features=categorical_features, 
        label='salary', 
        training=True
    )
    
    # Verify outputs are numpy arrays
    assert isinstance(X_processed, np.ndarray), "X_processed should be numpy array"
    assert isinstance(y_processed, np.ndarray), "y_processed should be numpy array"
    
    # Verify shapes make sense
    assert X_processed.shape[0] == 3, f"Should have 3 rows, got {X_processed.shape[0]}"
    assert len(y_processed) == 3, f"Should have 3 labels, got {len(y_processed)}"
    
    # Verify encoder and label binarizer are created
    assert encoder is not None, "Encoder should not be None in training mode"
    assert lb is not None, "Label binarizer should not be None in training mode"
    
    # Test inference mode (using the trained encoder and lb)
    test_inference_data = pd.DataFrame({
        'age': [40],
        'education': ['Bachelors'],
        'salary': ['>50K']
    })
    
    X_inf, y_inf, encoder_inf, lb_inf = process_data(
        test_inference_data,
        categorical_features=categorical_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Verify inference mode returns the same encoder and lb
    assert encoder_inf is encoder, "Should return the same encoder in inference mode"
    assert lb_inf is lb, "Should return the same label binarizer in inference mode"
