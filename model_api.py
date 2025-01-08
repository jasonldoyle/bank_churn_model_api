import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # Import the CORS module

model = None

def load_model():
    global model
    # Use joblib to load the model
    model = joblib.load('best_rf_model.pkl')

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/api/test', methods = ['POST'])
def test():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'no json data received'}), 400
    
    name = data.get('name')
    

    return jsonify({'message': f'Hello, {name}!'})
        

@app.route('/predict', methods=['POST'])  # Without trailing slash
def get_prediction():
    if request.method == 'POST':
        try:
            data = request.get_json()  # Get data posted as JSON
            df = pd.DataFrame(data)  # Convert to DataFrame
            predictions = model.predict(df)  # Make predictions
            return {"predictions": predictions.tolist()}
        except Exception as e:
            return {"error": str(e)}, 400
    elif request.method == 'OPTIONS':
        return '', 200  # Handle CORS preflight request

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=True)
