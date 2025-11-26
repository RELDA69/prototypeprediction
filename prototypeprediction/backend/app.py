from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from model import preprocess_input  # Import preprocessing function from model.py

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Preprocess input to match training data
        processed_data = preprocess_input(data)
        # Make prediction
        prediction = model.predict([processed_data])[0]
        # Decode prediction back to major name
        le_major = joblib.load('le_major.pkl')
        major_name = le_major.inverse_transform([prediction])[0]
        return jsonify({'prediction': major_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
