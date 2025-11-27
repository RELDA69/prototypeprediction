from flask import Blueprint, request, jsonify
import numpy as np
import traceback

def create_api_blueprint(model):
    api = Blueprint('api', __name__)

    @api.route('/favicon.ico')
    def favicon():
        return ('', 204)

    @api.route('/predict', methods=['POST', 'OPTIONS'])
    def predict():
        if request.method == 'OPTIONS':
            return ('', 200)

        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({'error': f'Invalid JSON: {e}'}), 400

        if not isinstance(data, dict):
            return jsonify({'error': 'Expected JSON object'}), 400

        try:
            # Case 1: frontend sends { "features": [...] }
            if 'features' in data:
                features = np.array(data['features'], dtype=float).reshape(1, -1)

            # Case 2: frontend sends individual keys
            else:
                required = [
                    'strongest_subjects', 'preferred_task',
                    'programming_skills', 'interest_in_technology',
                    'future_career_goal', 'preferred_work_type',
                    'preferred_thinking_style'
                ]
                missing = [k for k in required if k not in data]
                if missing:
                    return jsonify({'error': f'Missing required fields: {missing}'}), 400

                # Example mapping — adjust to match your training CSV column order
                strongest = data['strongest_subjects']
                if isinstance(strongest, str):
                    strongest = [strongest]

                features = [
                    1 if "Math" in strongest else 0,
                    1 if "Science" in strongest else 0,
                    int(data['programming_skills']),
                    int(data['interest_in_technology']),
                    # You’ll need to encode categorical fields consistently with training
                    # For now, just include them as strings or simple flags
                ]

                features = np.array(features, dtype=float).reshape(1, -1)

            prediction = model.predict(features)
            return jsonify({'prediction': str(prediction[0])}), 200

        except Exception as e:
            print("ERROR in /predict:", traceback.format_exc())
            return jsonify({'error': f'Prediction failed: {e}'}), 500

    return api