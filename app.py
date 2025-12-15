from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Run train_model.py first.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        # Extract features from the request
        # Expected format: {"n_lights": 5, "w_lights": 10, ...}

        # Helper to safely get values, default to 0 if missing
        def get_val(key, type_cast=float):
            return type_cast(data.get(key, 0))

        features = [
            get_val('n_lights', int), get_val('w_lights'),
            get_val('n_fans', int), get_val('w_fans'),
            get_val('n_ac', int), get_val('w_ac'),
            get_val('n_tv', int), get_val('w_tv'),
            get_val('n_fridge', int), get_val('w_fridge')
        ]

        unit_price = get_val('unit_price', float)

        # Reshape for prediction
        input_data = np.array([features])

        # Make prediction (Daily kWh)
        daily_kwh = model.predict(input_data)[0]

        # Ensure non-negative
        daily_kwh = max(0, daily_kwh)

        weekly_kwh = daily_kwh * 7
        monthly_kwh = daily_kwh * 30

        estimated_bill = monthly_kwh * unit_price

        return jsonify({
            'success': True,
            'daily_kwh': float(daily_kwh),
            'weekly_kwh': float(weekly_kwh),
            'monthly_kwh': float(monthly_kwh),
            'estimated_bill': float(estimated_bill)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
