import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('data_cleaned.csv')

# 2. Define Features (X) and Target (y)
# We want to predict PCE using VOC, JSC, and FF
X = df[['VOC', 'JSC', 'FF']]
y = df['PCE']

# 3. Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Example Prediction
new_data = [[1400, 6.46, 0.792]] # VOC, JSC, FF
print(f"Predicted PCE for {new_data}: {model.predict(new_data)[0]:.2f}")