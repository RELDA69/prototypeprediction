import re
from pathlib import Path
import pickle

MODELS_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODELS_DIR / 'model_pipeline.pkl'
ENCODED_FEATURES_PATH = MODELS_DIR / 'encoded_features.pkl'

_model = None
_encoded_features = None

def _normalize_text(s):
    if s is None:
        return ''
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s

def _load():
    global _model, _encoded_features
    if _model is not None:
        return _model, _encoded_features
    try:
        import joblib
    except Exception as e:
        raise ImportError(f"joblib is required: {e}")
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    else:
        _model = None
    if ENCODED_FEATURES_PATH.exists():
        with open(ENCODED_FEATURES_PATH, 'rb') as f:
            _encoded_features = pickle.load(f)
    else:
        _encoded_features = None
    return _model, _encoded_features

def _build_encoded_row(encoded_features, strongest_subjects, preferred_task, programming_skills, interest_in_technology, future_career_goal, preferred_work_type, preferred_thinking_style):
    import pandas as pd
    row = {c: 0 for c in encoded_features}
    if 'programming_skills' in row:
        try:
            row['programming_skills'] = float(programming_skills or 0)
        except Exception:
            row['programming_skills'] = 0.0
    if 'interest_in_technology' in row:
        try:
            row['interest_in_technology'] = float(interest_in_technology or 0)
        except Exception:
            row['interest_in_technology'] = 0.0
    norm_features = {c: _normalize_text(c) for c in encoded_features}
    mapping_inputs = {'preferred_task': preferred_task, 'future_career_goal': future_career_goal, 'preferred_work_type': preferred_work_type, 'preferred_thinking_style': preferred_thinking_style}
    for _, value in mapping_inputs.items():
        if not value:
            continue
        nv = _normalize_text(value)
        for col, ncol in norm_features.items():
            if nv and nv in ncol:
                row[col] = 1
    subjects = []
    if isinstance(strongest_subjects, list):
        subjects = strongest_subjects
    elif isinstance(strongest_subjects, str):
        if ';' in strongest_subjects:
            subjects = [s.strip() for s in strongest_subjects.split(';') if s.strip()]
        elif ',' in strongest_subjects:
            subjects = [s.strip() for s in strongest_subjects.split(',') if s.strip()]
        else:
            subjects = [strongest_subjects.strip()]
    for s in subjects:
        ns = _normalize_text(s)
        for col, ncol in norm_features.items():
            if ns and ns in ncol:
                row[col] = 1
    return pd.DataFrame([row], columns=encoded_features)

def predict_major(strongest_subjects, preferred_task, programming_skills, interest_in_technology, future_career_goal, preferred_work_type, preferred_thinking_style):
    import pandas as pd
    model, encoded_features = _load()
    if model is None:
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run model.py to train.")
    if hasattr(model, 'named_steps') and 'pre' in model.named_steps:
        ss = ';'.join(str(s).strip() for s in (strongest_subjects or [])) if isinstance(strongest_subjects, list) else str(strongest_subjects or '')
        row = pd.DataFrame([{'strongest_subjects': ss, 'preferred_task': preferred_task, 'programming_skills': programming_skills, 'interest_in_technology': interest_in_technology, 'future_career_goal': future_career_goal, 'preferred_work_type': preferred_work_type, 'preferred_thinking_style': preferred_thinking_style}])
        pred = model.predict(row)
        return str(pred[0])
    if encoded_features is None:
        raise RuntimeError("Encoded model loaded but metadata missing. Retrain with model.py")
    encoded_row = _build_encoded_row(encoded_features, strongest_subjects, preferred_task, programming_skills, interest_in_technology, future_career_goal, preferred_work_type, preferred_thinking_style)
    encoded_row = encoded_row.reindex(columns=encoded_features, fill_value=0)
    pred = model.predict(encoded_row)
    return str(pred[0])
