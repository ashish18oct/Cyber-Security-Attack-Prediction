import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "cybersecurity_attacks.csv" 
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Loaded Successfully!")
print("\n Basic Dataset Information:")
print(df.info())

# Display first few rows
display(df.head())

# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("\n Missing Values in Dataset:")
print(missing_values)

# Check class distribution
print("\n Attack Type Distribution:")
print(df["Attack Type"].value_counts())

# Display summary statistics
print("\n Summary Statistics:")
display(df.describe())

# Display feature types (categorical vs numerical)
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
print("\n Numerical Features:", numerical_features)
print("\n Categorical Features:", categorical_features)
