# task1.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load dataset from local file
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("data.csv")

# Step 2: Replace 0s with NaN in medical columns
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Step 3: Fill NaN values with median
df.fillna(df.median(), inplace=True)

# Step 4: Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 5: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Save processed data
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "processed_data.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Task 1 Complete: Data processed and saved.")
