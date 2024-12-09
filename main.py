import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load Dataset
data = pd.read_csv("loan_sanction_test.csv")
print("Dataset Loaded Successfully")
print(data.head())

# Statistical Analysis
print("Statistical Analysis:")
print(data.describe())

# Check for null values or missing data
missing_data = data.isnull().sum()
print(f"Columns with missing values:\n{missing_data[missing_data > 0]}")

# Separate categorical and numerical columns
cat_columns = data.select_dtypes(include="object").columns
num_columns = data.select_dtypes(include="number").columns

print(f"Categorical columns: {list(cat_columns)}")
print(f"Numerical columns: {list(num_columns)}")

# Fill missing values
for col in num_columns:
    data[col].fillna(data[col].mean(), inplace=True)

for col in cat_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Confirm no missing values remain
print("Missing values after imputation:")
print(data.isnull().sum())

# Apply Label Encoding to all categorical columns
label_encoders = {}
for col in cat_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for future use

# Feature-Target Split
X = data.drop('Property_Area', axis=1)  # Predicting 'Property_Area'
y = data['Property_Area']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Scaler
os.makedirs('model', exist_ok=True)
model_path = 'model/home_loan_sanction_model.pkl'
scaler_path = 'model/scaler.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
