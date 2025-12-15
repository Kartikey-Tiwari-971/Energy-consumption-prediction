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
        # Expected format: {"temperature": 25, "humidity": 60, "square_footage": 1500, "occupancy": 3, "hour_of_day": 14}
        features = [
            float(data['temperature']),
            float(data['humidity']),
            float(data['square_footage']),
            int(data['occupancy']),
            int(data['hour_of_day'])
        ]

        # Reshape for prediction
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({
            'success': True,
            'prediction': float(prediction)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
