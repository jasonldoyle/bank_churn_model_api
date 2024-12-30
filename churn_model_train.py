import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import joblib

df = pd.read_csv('/Users/jason/Desktop/Github/Bank Churn Model/Dataset/Churn_Modelling_Dataset.csv')

df = df.drop(columns=['RowNumber', 'Surname', 'CustomerId'])

df.columns = [
    'credit_score',
    'geography',
    'gender',
    'age',
    'tenure',
    'balance',
    'num_of_products',
    'has_cr_card',
    'is_active_member',
    'estimated_salary',
    'exited'
]

def label_encode_cat_features(df, cat_features):
    for feature in cat_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    return df

cat_feats = ['geography', 'gender', 'has_cr_card', 'is_active_member']
df = label_encode_cat_features(df, cat_feats)

X = df.drop(columns=['exited'])  
y = df['exited']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Define best parameters for the Random Forest model
best_params = {
    'bootstrap': True,
    'max_depth': 7,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 50
}

# Initialize and train the Random Forest model
rf = RandomForestClassifier(
    bootstrap=best_params['bootstrap'], 
    max_depth=best_params['max_depth'], 
    n_estimators=best_params['n_estimators'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split']
)

rf.fit(X_train, y_train)

# Make predictions
y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)

# Evaluate the model
conf_matrix = confusion_matrix(y_val, y_val_pred)
class_report = classification_report(y_val, y_val_pred)

print(f"Accuracy on training set: {rf.score(X_train, y_train):.3f}")
print(f"Accuracy on validation set: {rf.score(X_val, y_val):.3f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save the trained model to a file
model_filename = "best_rf_model.pkl"
joblib.dump(rf, model_filename)
print(f"Trained model saved as {model_filename}")

# Create a DataFrame to compare actual vs predicted values
comparison_df = pd.DataFrame({
    'Actual': y_val.values,
    'Predicted': y_val_pred,
    'Difference': y_val_pred - y_val.values,
})

# Display the first few rows of the comparison DataFrame
print("\nComparison DataFrame (first 5 rows):")
print(comparison_df.head())

# Load the model
joblib.dump(rf, "best_rf_model.pkl")
print("Model saved as 'best_rf_model.pkl'")