#!/usr/bin/env python3
"""
Census Data Exploration Script
Explores the structure and characteristics of the census.csv dataset
"""

import pandas as pd
import numpy as np

# Load the data
print("Loading census data...")
df = pd.read_csv('data/census.csv')

print("="*60)
print("CENSUS DATASET OVERVIEW")
print("="*60)

# Basic info
print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe(include='all'))

print("\n" + "="*60)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*60)

# Identify categorical vs numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"  - {col}")

print(f"\nNumerical columns ({len(numerical_cols)}):")
for col in numerical_cols:
    print(f"  - {col}")

# Examine each categorical column
print("\nCategorical Features Details:")
for col in categorical_cols:
    unique_values = df[col].nunique()
    top_values = df[col].value_counts().head(3)
    print(f"\n{col}:")
    print(f"  Unique values: {unique_values}")
    print(f"  Top 3 values:")
    for val, count in top_values.items():
        percentage = (count / len(df)) * 100
        print(f"    '{val}': {count:,} ({percentage:.1f}%)")

print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS")
print("="*60)

# Analyze the target variable (assuming it's 'salary')
if 'salary' in df.columns:
    target_col = 'salary'
    print(f"\nTarget variable: {target_col}")
    print(df[target_col].value_counts())
    print(f"\nTarget distribution:")
    target_counts = df[target_col].value_counts()
    for val, count in target_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {val}: {count:,} ({percentage:.1f}%)")

print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)

missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    print("\nMissing values found:")
    for col, count in missing_data.items():
        percentage = (count / len(df)) * 100
        print(f"  {col}: {count:,} ({percentage:.2f}%)")
else:
    print("\n✅ No missing values found!")

# Check for potential missing value indicators
print("\nChecking for potential missing value indicators ('?', 'Unknown', etc.):")
for col in categorical_cols:
    suspicious_values = ['?', 'Unknown', 'unknown', ' ?', '? ', 'NA', 'N/A']
    for val in suspicious_values:
        count = (df[col] == val).sum()
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"  {col} has {count:,} '{val}' values ({percentage:.2f}%)")

print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nFirst 5 rows:")
print(df.head())

print("\nRandom 3 rows:")
print(df.sample(3, random_state=42))

print("\n" + "="*60)
print("EXPLORATION COMPLETE!")
print("="*60)
