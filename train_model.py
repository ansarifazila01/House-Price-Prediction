
import pandas as pd
import json
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Load Indian housing dataset
with open("data/price.json", "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Keep required columns
df = df[["bedrooms", "location", "area", "price"]].copy()

# Clean data
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df.dropna(subset=["bedrooms", "location", "area", "price"])

df = df[
    (df["area"] > 0) &
    (df["price"] > 0)
]

print("Cleaned dataset shape:", df.shape)

# Features and target
X = df[["bedrooms", "location", "area"]]
y = df["price"]

# Preprocessing
categorical_features = ["bedrooms", "location"]
numeric_features = ["area"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("numeric", SimpleImputer(strategy="median"),
         numeric_features)
    ]
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Training Completed!")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model
os.makedirs("data", exist_ok=True)

with open("data/house_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nIndian House Price Model Saved Successfully!")
print("Price unit: Dataset price units (likely lakh INR)")
