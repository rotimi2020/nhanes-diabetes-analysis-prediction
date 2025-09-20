#!/usr/bin/env python
"""
Script to train and save the diabetes prediction model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

# Load the dataset
df = pd.read_csv('ml_ready_nhanes.csv')

# Note :

# Mapping dictionaries
# Gender_Code = {1: 'Male', 2: 'Female'}

# Race_Code = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 7: 'Other Race'}

# Education_Code_Imputed = {1: 'Less than 9th grade', 2: '9-11th grade', 3: 'High school graduate',4: 'Some college or AA degree', 5: 'College graduate'}

# Family_Diabetes_Code_Imputed =  {1: 'Yes', 2: 'No'}

# Risk_Level	= { 0 : 'High Risk' , 1 : 'Low Risk'}

# Obesity_Status = {0:'Non-Obese' , 1: 'Obese' ,2 : 'Overweight'}

# Diabetes_Status = {0:'No Diabetes' , 1: 'Obese' ,2 : 'Diabetes'}
# Diabetes_Status is the target

# Prepare features and target
X = df.drop(columns='Diabetes_Status')
y = df['Diabetes_Status'].map({'No Diabetes': 0, 'Diabetes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LGBMClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.01,
    random_state=42,
    class_weight="balanced",
    boosting_type="gbdt",
    verbose=-1
)
model.fit(X_train_scaled, y_train)
print(X.columns)

# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

print("Model training complete and files saved!")