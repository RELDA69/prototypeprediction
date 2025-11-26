import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / 'models'
MODEL_OUT = MODELS_DIR / 'model_pipeline.pkl'

# Use the merged dataset with majors
DATA_FILE = BACKEND_DIR.parent / 'data'/ 'train_with_major.csv'

class StrongestSubjectsBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sep=';'):
        self.sep = sep
        self.subjects_ = []

    def fit(self, X, y=None):
        series = X['strongest_subjects'].fillna('').apply(lambda v: v if isinstance(v, list) else str(v))
        sets = series.apply(lambda s: [i.strip() for i in s.split(self.sep) if i.strip()])
        unique = set()
        for lst in sets:
            unique.update(lst)
        self.subjects_ = sorted(unique)
        return self

    def transform(self, X):
        series = X['strongest_subjects'].fillna('').apply(lambda v: v if isinstance(v, list) else str(v))
        sets = series.apply(lambda s: [i.strip() for i in s.split(self.sep) if i.strip()])
        out = np.zeros((len(sets), len(self.subjects_)), dtype=int)
        for i, lst in enumerate(sets):
            for item in lst:
                if item in self.subjects_:
                    out[i, self.subjects_.index(item)] = 1
        return out

def train():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded: {DATA_FILE}")

    if 'major' not in df.columns:
        raise RuntimeError("CSV must contain 'major' column as target label.")

    # Features vs target
    X = df.drop(columns=['major'])
    y = df['major'].astype(str)

    # Preprocessing: detect categorical/numeric columns
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numeric_cols = [col for col in X.columns if np.issubdtype(X[col].dtype, np.number)]

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ], remainder='drop')

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"✅ Pipeline model trained and saved to {MODEL_OUT}")

if __name__ == '__main__':
    train()