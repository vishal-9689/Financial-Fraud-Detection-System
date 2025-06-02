from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler
try:
    model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/amount_scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    exit(1)

# Define the route for checking fraud
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the incoming JSON request
        data = request.get_json()

        # Check if necessary features are present
        required_features = ['Amount'] + [f'V{i}' for i in range(1, 29)]
        if not all(feature in data for feature in required_features):
            return jsonify({'error': f'Missing required features: {set(required_features) - set(data.keys())}'}), 400

        # Prepare input data
        transaction_data = np.array([data[feature] for feature in required_features]).reshape(1, -1)

        # Normalize the 'Amount' using the loaded scaler
        transaction_data[0][0] = scaler.transform([[data['Amount']]])[0][0]

        # Make prediction using the model
        proba = model.predict_proba(transaction_data)[0][1]

        # Determine the result based on the probability
        if proba > 0.7:
            result = 'High Fraud Risk'
        elif proba > 0.3:
            result = 'Moderate Fraud Risk'
        else:
            result = 'Low Fraud Risk'

        # Return the result as a JSON response
        return jsonify({
            'result': result,
            'fraud_probability': proba
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
