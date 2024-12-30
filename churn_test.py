import joblib
import pandas as pd

rf_loaded = joblib.load("best_rf_model.pkl")

X_new = pd.DataFrame({
    'credit_score': [600, 750],
    'geography': [1, 0],  
    'gender': [1, 0],  
    'age': [40, 55],
    'tenure': [5, 3],
    'balance': [50000, 60000],
    'num_of_products': [2, 1],
    'has_cr_card': [1, 2],
    'is_active_member': [1, 0],
    'estimated_salary': [70000, 80000]
})

new_predictions = rf_loaded.predict(X_new)
print("Predictions for new data:", new_predictions)

