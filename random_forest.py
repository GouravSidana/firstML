import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your specific dataset
df = pd.read_csv('ML_Ready_Dataset.csv')

# 2. Define Features (X) and Target (y)
X = df[['VOC', 'JSC', 'FF']]
y = df['PCE']

# 3. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the Random Forest Model
# n_estimators=100 means the "forest" is made of 100 individual decision trees
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions on the 20% test data
y_pred = model.predict(X_test)

# 6. Evaluate the Model's Performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("--- Random Forest Model Results ---")
print(f"R-squared Score: {r2:.4f} (Closer to 1.0 is better)")
print(f"Mean Squared Error: {mse:.4f} (Lower is better)")

# 7. Feature Importance
# Random Forest can mathematically tell us which physical property drives PCE the most!
importances = model.feature_importances_
print(f"VOC Importance: {importances[0] * 100:.2f}%")
print(f"JSC Importance: {importances[1] * 100:.2f}%")
print(f"FF Importance:  {importances[2] * 100:.2f}%")

# Example Prediction
new_data = [[1400, 6.46, 0.792]] # VOC, JSC, FF
print(f"Predicted PCE for {new_data}: {model.predict(new_data)[0]:.2f}")