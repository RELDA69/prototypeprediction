import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / 'models'
MODEL_OUT = MODELS_DIR / 'model_pipeline.pkl'
DATA_FILE = BACKEND_DIR.parent / "data" / "DMA-DATASET.csv"

def train():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    # Load Excel dataset
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded: {DATA_FILE}")

    # Target column: Current Course (major)
    y = df["Current Course"]

    # Drop identifiers and target
    X = df.drop(columns=["Name", "email_id", "Current Course"])

    # Define categorical and numeric columns
    categorical_cols = [
    "Year",  # Note the trailing space
    "Technical Skills",
    "Programming Languages",  # Note the leading spaces
    "Soft Skills",
    "Projects",  # Note the leading space
    "Career Interest",
    "Challenges",  # Note the trailing space
    "Support required",
    "Method"  # Note the leading space
    ]
    numeric_cols = ["Programming Languages Ratings", "Soft Skills Rating"]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='drop'
    )

    # Pipeline
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    # Train
    pipeline.fit(X, y)

    # Save pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"✅ Pipeline model trained and saved to {MODEL_OUT}")

if __name__ == '__main__':
    train()