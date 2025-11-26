from flask import Blueprint, request, jsonify
from models.recommender import predict_major

api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract data from the request
        strongest_subjects = data.get('strongest_subjects')
        preferred_task = data.get('preferred_task')
        programming_skills = data.get('programming_skills')
        interest_in_technology = data.get('interest_in_technology')
        future_career_goal = data.get('future_career_goal')
        preferred_work_type = data.get('preferred_work_type')
        preferred_thinking_style = data.get('preferred_thinking_style')

        # Call the prediction function
        recommended_major = predict_major(
            strongest_subjects,
            preferred_task,
            programming_skills,
            interest_in_technology,
            future_career_goal,
            preferred_work_type,
            preferred_thinking_style
        )

        # Return format expected by frontend: { major: "Recommended Major" }
        return jsonify({'major': recommended_major}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400