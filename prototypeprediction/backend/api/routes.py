from flask import Blueprint, request, jsonify
from models.recommender import predict_major
import traceback

api = Blueprint('api', __name__)

_LEGACY_MAP = {
    'task_enjoy': 'preferred_task',
    'prog_skills': 'programming_skills',
    'tech_interest': 'interest_in_technology',
    'career_want': 'future_career_goal',
    'prefer_work': 'preferred_work_type',
    'prefer_creative_logical': 'preferred_thinking_style'
}

@api.route('/favicon.ico')
def favicon():
    return ('', 204)

@api.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return ('', 200)

    try:
        raw = request.get_data()
        print("---- RAW REQUEST BODY (bytes) ----")
        print(raw)
        data = request.get_json(force=True)
    except Exception as e:
        body_preview = raw.decode('utf-8', errors='replace') if 'raw' in locals() else '<no body>'
        print("JSON parse error:", e)
        print("Body preview:", body_preview)
        return jsonify({'error': f'Invalid JSON or Content-Type. parse error: {e}', 'body_preview': body_preview}), 400

    if not isinstance(data, dict):
        return jsonify({'error': 'Expected JSON object'}), 400

    # Map legacy keys -> new keys only if new key missing
    for old, new in _LEGACY_MAP.items():
        if old in data and new not in data:
            data[new] = data[old]

    print("DEBUG /predict received keys:", list(data.keys()))
    print("DEBUG /predict full payload:", data)

    required = [
        'strongest_subjects', 'preferred_task',
        'programming_skills', 'interest_in_technology',
        'future_career_goal', 'preferred_work_type',
        'preferred_thinking_style'
    ]
    missing = [k for k in required if k not in data or data.get(k) in (None, '')]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        prog = int(data.get('programming_skills'))
        interest = int(data.get('interest_in_technology'))
    except Exception:
        return jsonify({'error': "programming_skills and interest_in_technology must be integers"}), 400

    strongest = data.get('strongest_subjects')
    if isinstance(strongest, str):
        if ';' in strongest:
            strongest = [s.strip() for s in strongest.split(';') if s.strip()]
        elif ',' in strongest:
            strongest = [s.strip() for s in strongest.split(',') if s.strip()]
        else:
            strongest = [strongest.strip()]
    elif not isinstance(strongest, list):
        return jsonify({'error': 'strongest_subjects must be an array or delimited string'}), 400

    try:
        recommended_major = predict_major(
            strongest,
            data.get('preferred_task'),
            prog,
            interest,
            data.get('future_career_goal'),
            data.get('preferred_work_type'),
            data.get('preferred_thinking_style')
        )
    except KeyError as ke:
        print("Missing key used inside code:", ke)
        return jsonify({'error': f"Missing key inside server code: {ke}"}), 400
    except Exception as e:
        print("ERROR in predict_major:", traceback.format_exc())
        return jsonify({'error': str(e)}), 400

    return jsonify({'major': recommended_major}), 200