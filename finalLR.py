import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD DATA & SPLIT (80-20)
df = pd.read_csv('ML_Ready_Dataset.csv')

X = df[['VOC', 'JSC', 'FF']]
y = df['PCE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TRAIN THE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 3. CALCULATE & PRINT TEXT OUTPUTS
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
coefficients = model.coef_

print("      LINEAR REGRESSION MODEL RESULTS")
print(f"R-squared Score    : {r2:.4f} (Closer to 1.0 is better)")
print(f"Mean Squared Error : {mse:.4f} (Lower is better)")
print("FEATURE COEFFICIENTS (Math Weights):")
print(f"VOC Coefficient    : {coefficients[0]:.6f}")
print(f"JSC Coefficient    : {coefficients[1]:.6f}")
print(f"FF  Coefficient    : {coefficients[2]:.6f}")
print("Baseline (Intercept):", round(model.intercept_, 4))

# 4. GENERATE SEPARATE VISUAL PLOTS
sns.set_theme(style="whitegrid")

#Coefficient Bar Chart
plt.figure(figsize=(8, 6))
sns.barplot(x=coefficients, y=X.columns, palette="coolwarm")
plt.title('Linear Regression: Feature Coefficients', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value (Impact on PCE)', fontsize=12)
plt.ylabel('Physical Properties', fontsize=12)
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=300)
plt.close() 

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='#11caa0', s=60, label='Linear Model Predictions')

# Draw the Red "Perfect Score" Line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Score (y=x)')

plt.title('Linear Regression Accuracy: Actual vs. Predicted PCE', fontsize=14, fontweight='bold')
plt.xlabel('Actual Experimental PCE', fontsize=12)
plt.ylabel('Model Predicted PCE', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('lr_actual_vs_predicted.png', dpi=300)
plt.close()

print("\nSuccess! Linear Regression plots saved separately as:")
print("1. 'lr_coefficients.png'")
print("2. 'lr_actual_vs_predicted.png'")