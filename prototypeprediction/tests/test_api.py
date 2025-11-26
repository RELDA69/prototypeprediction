from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('backend/ml/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert input data to DataFrame for prediction
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return jsonify({'recommended_major': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
