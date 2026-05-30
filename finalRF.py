import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 1. LOAD DATA & SPLIT (80-20)
df = pd.read_csv('data_cleaned.csv')

X = df[['VOC', 'JSC', 'FF']]
y = df['PCE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. TRAIN THE RANDOM FOREST MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 3. CALCULATE & PRINT TEXT OUTPUTS
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
importances = model.feature_importances_

print("       RANDOM FOREST MODEL RESULTS")
print(f"R-squared Score    : {r2:.4f} (Closer to 1.0 is better)")
print(f"Mean Squared Error : {mse:.4f} (Lower is better)")
print("FEATURE IMPORTANCES (What drives efficiency?):")
print(f"VOC (Voltage)      : {importances[0] * 100:.2f}%")
print(f"JSC (Current)      : {importances[1] * 100:.2f}%")
print(f"FF  (Fill Factor)  : {importances[2] * 100:.2f}%")

# 4. GENERATE SEPARATE VISUAL PLOTS
sns.set_theme(style="whitegrid")

# Feature Importance Bar Chart
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=X.columns, palette="viridis")
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance (0.0 to 1.0)', fontsize=12)
plt.ylabel('Physical Properties', fontsize=12)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300)
plt.close() # Close canvas so it doesn't mix with the next plot

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='#005088', s=60, label='Model Predictions')

# Draw the Red "Perfect Score" Line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Score (y=x)')

plt.title('Accuracy: Actual vs. Predicted PCE', fontsize=14, fontweight='bold')
plt.xlabel('Actual Experimental PCE', fontsize=12)
plt.ylabel('Model Predicted PCE', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('rf_actual_vs_predicted.png', dpi=300)
plt.close() # Close canvas

print("\nSuccess! Plots saved separately as:")
print("1. 'rf_feature_importance.png'")
print("2. 'rf_actual_vs_predicted.png'")