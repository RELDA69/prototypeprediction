from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('ml/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Prepare the input data for prediction
    input_data = pd.DataFrame([{
        'strongest_subjects': data['strongest_subjects'],
        'preferred_task': data['preferred_task'],
        'programming_skills': data['programming_skills'],
        'interest_in_technology': data['interest_in_technology'],
        'future_career_goal': data['future_career_goal'],
        'preferred_work_type': data['preferred_work_type'],
        'preferred_thinking_style': data['preferred_thinking_style']
    }])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return jsonify({'recommended_major': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)