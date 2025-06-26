"""
Script to train machine learning model on Census data.

This script loads the census data, preprocesses it, trains a model,
evaluates performance, and saves the trained model and artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice
)


def main():
    """Main function to execute the ML pipeline."""
    
    print("Starting ML Pipeline...")
    print("=" * 50)
    
    # Load the census data
    print("Loading census data...")
    data = pd.read_csv("data/census.csv")
    print(f"Data loaded: {data.shape[0]:,} rows, {data.shape[1]} columns")
    
    # Handle missing values (replace '?' with most frequent value for each column)
    print("Handling missing values...")
    categorical_columns = ['workclass', 'occupation', 'native-country']
    for col in categorical_columns:
        if '?' in data[col].values:
            # Replace '?' with the most frequent value
            most_frequent = data[col].mode()[0]
            data[col] = data[col].replace('?', most_frequent)
            print(f"  Replaced '?' in {col} with '{most_frequent}'")
    
    # Define categorical features (all object columns except target)
    categorical_features = [
        "workclass",
        "education", 
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    
    print(f"Categorical features: {categorical_features}")
    
    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(
        data, 
        test_size=0.20, 
        random_state=42,
        stratify=data['salary']  # Maintain class distribution
    )
    print(f"Train set: {train_data.shape[0]:,} samples")
    print(f"Test set: {test_data.shape[0]:,} samples")
    
    # Process the training data
    print("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    print(f"Training data processed: X shape {X_train.shape}, y shape {y_train.shape}")
    
    # Process the test data with the fitted encoder and label binarizer
    print("Processing test data...")
    X_test, y_test, _, _ = process_data(
        test_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )
    print(f"Test data processed: X shape {X_test.shape}, y shape {y_test.shape}")
    
    # Train the model
    print("Training model...")
    model = train_model(X_train, y_train)
    print("Model training completed!")
    
    # Save the model and encoders
    print("Saving model and encoders...")
    save_model(model, "model/model.pkl")
    save_model(encoder, "model/encoder.pkl")
    save_model(lb, "model/lb.pkl")
    print("Model saved to model/model.pkl")
    print("Encoder saved to model/encoder.pkl")
    print("Label binarizer saved to model/lb.pkl")
    
    # Test loading the model
    print("Loading model from model/model.pkl")
    loaded_model = load_model("model/model.pkl")
    
    # Run inference on test data
    print("Running inference on test data...")
    predictions = inference(loaded_model, X_test)
    
    # Compute overall performance metrics
    print("Computing performance metrics...")
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    print(f"Overall Performance:")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")
    
    # Compute performance on categorical slices
    print("Computing performance on categorical slices...")
    slice_output = []
    slice_output.append("Model Performance on Categorical Slices")
    slice_output.append("=" * 50)
    slice_output.append(f"Overall Performance:")
    slice_output.append(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")
    slice_output.append("")
    
    for feature in categorical_features:
        print(f"Computing slices for feature: {feature}")
        slice_output.append(f"Feature: {feature}")
        slice_output.append("-" * 30)
        
        # Get unique values for this feature
        unique_values = test_data[feature].unique()
        
        for value in unique_values:
            # Count how many samples have this value
            count = len(test_data[test_data[feature] == value])
            
            if count > 0:  # Only process if there are samples
                try:
                    # Compute performance for this slice
                    slice_precision, slice_recall, slice_fbeta = performance_on_categorical_slice(
                        test_data,
                        feature,
                        value,
                        categorical_features,
                        "salary",
                        encoder,
                        lb,
                        loaded_model
                    )
                    
                    output_line = f"Precision: {slice_precision:.4f} | Recall: {slice_recall:.4f} | F1: {slice_fbeta:.4f}"
                    count_line = f"{feature}: {value}, Count: {count:,}"
                    
                    print(f"  {output_line}")
                    print(f"  {count_line}")
                    
                    slice_output.append(output_line)
                    slice_output.append(count_line)
                    
                except Exception as e:
                    print(f"  Error processing {feature}={value}: {e}")
                    slice_output.append(f"Error processing {feature}={value}: {e}")
        
        slice_output.append("")  # Add blank line between features
    
    # Save slice output to file
    print("Saving slice output to slice_output.txt...")
    with open("slice_output.txt", "w") as f:
        f.write("\n".join(slice_output))
    print("Slice output saved to slice_output.txt")
    
    print("=" * 50)
    print("ML Pipeline completed successfully!")
    print(f"Final model performance - Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")


if __name__ == "__main__":
    main()
