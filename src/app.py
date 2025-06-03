from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and scaler once on startup
try:
    model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/amount_scaler.pkl')
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    exit(1)

# Maps for categorical encoding
transaction_type_map = {"Online": 0, "In-Store": 1, "ATM": 2}
card_type_map = {"Credit": 0, "Debit": 1, "Prepaid": 2}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['Transaction_Type', 'Card_Type']
        missing = [f for f in required_features if f not in data]
        if missing:
            return jsonify({'error': f'Missing required features: {missing}'}), 400

        # Validate numeric features
        for f in ['Amount'] + [f'V{i}' for i in range(1, 29)]:
            try:
                float(data[f])
            except (ValueError, TypeError):
                return jsonify({'error': f"Feature '{f}' must be numeric."}), 400

        # Encode categorical features
        try:
            transaction_type_encoded = transaction_type_map[data['Transaction_Type']]
            card_type_encoded = card_type_map[data['Card_Type']]
        except KeyError as e:
            return jsonify({'error': f'Invalid category value: {str(e)}'}), 400

        # Prepare feature array in correct order
        values = [float(data['Amount'])] + [float(data[f'V{i}']) for i in range(1, 29)] + [transaction_type_encoded, card_type_encoded]
        input_array = np.array(values).reshape(1, -1)

        # Scale 'Amount' feature
        input_array[0][0] = scaler.transform([[values[0]]])[0][0]

        # Predict fraud probability
        proba = model.predict_proba(input_array)[0][1]

        # Risk categories based on probability thresholds
        if proba > 0.7:
            result = 'High Fraud Risk'
        elif proba > 0.3:
            result = 'Moderate Fraud Risk'
        else:
            result = 'Low Fraud Risk'

        return jsonify({
            'result': result,
            'fraud_probability': proba
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
