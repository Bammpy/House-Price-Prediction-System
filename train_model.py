# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define expanded synthetic data with logical trends
data = {
    'bedrooms': [2, 3, 4, 1, 5, 2, 3, 4, 2, 5] * 5,
    'location': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5,
    'stories': [1, 2, 2, 1, 3, 1, 2, 3, 1, 2] * 5,
    'furnishings': [2, 1, 0, 1, 2, 0, 1, 2, 1, 0] * 5,
    'bathrooms': [1, 2, 3, 1, 4, 2, 2, 3, 1, 4] * 5,
    'guestroom': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0] * 5,
    'electricity': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1] * 5,
    'water': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1] * 5,
    'internet': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 5,
    'waste_disposal': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1] * 5,
    'proximity_amenities': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 5,
    'urbanization': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 5,
    'price': [55000, 65000, 95000, 35000, 120000, 50000, 70000, 100000, 48000, 110000] * 5
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['price'])
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save model and feature names as a dictionary
feature_names = X.columns.tolist()
background_data = X_train.head(10)  # Use first 10 rows of X_train as background
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump({'model': model, 'feature_names': feature_names, 'background_data': background_data}, file)

print("Model trained and saved as 'house_price_model.pkl' with feature names")

# Feature importance visualization
importances = model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.title("🔍 Feature Importance in House Price Prediction")
plt.show()

# Evaluate model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R^2 Score: {train_score:.2f}")
print(f"Testing R^2 Score: {test_score:.2f}")