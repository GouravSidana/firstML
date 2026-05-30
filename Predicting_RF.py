import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Ignore background warnings for a cleaner terminal output
warnings.filterwarnings('ignore', category=UserWarning)

# 1. Load your specific dataset
df = pd.read_csv('data_cleaned.csv')

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

# 8. INTERACTIVE PREDICTION FUNCTION
def predict_efficiency():

    print("   SOLAR CELL EFFICIENCY PREDICTOR")
    
    # This loop keeps the tool running until you tell it to stop
    while True:
        try:
            print("\nPlease enter your experimental values (or type 'q' to quit):")
            
            # Get VOC
            voc_input = input("1. Enter VOC (in mV) : ")
            if voc_input.lower() == 'q': break
            user_voc = float(voc_input)

            # Get JSC
            jsc_input = input("2. Enter JSC (in mA/cm²) : ")
            if jsc_input.lower() == 'q': break
            user_jsc = float(jsc_input)

            # Get FF
            ff_input = input("3. Enter FF : ")
            if ff_input.lower() == 'q': break
            user_ff = float(ff_input)

            # Package the inputs for the model
            new_data = [[user_voc, user_jsc, user_ff]]
            
            # Generate Prediction
            predicted_pce = model.predict(new_data)
            
            # Display Output
            print(f" PREDICTED PCE (Efficiency): {predicted_pce[0]:.2f}%")
            
        except ValueError:
            print("\n Error: Please enter valid numbers only!")

# Run the function
predict_efficiency()
print("\nPredictor closed. Good luck with your research!")