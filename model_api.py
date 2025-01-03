import pickle
import pandas as pd
import numpy as np
from flask import Flask, request
import joblib  # Replace pickle with joblib

model = None

def load_model():
    global model
    # Use joblib to load the model
    model = joblib.load('best_rf_model.pkl')

app = Flask(__name__)

@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        try:
            # Get JSON data and convert it to a Pandas DataFrame
            data = request.get_json()
            df = pd.DataFrame(data)
            
            # Make predictions
            predictions = model.predict(df)
            return {"predictions": predictions.tolist()}
        except Exception as e:
            return {"error": str(e)}, 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=True)