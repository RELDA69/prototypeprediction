from flask import Flask
from flask_cors import CORS
from pathlib import Path
import joblib

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Import your blueprint
from api.routes import api
app.register_blueprint(api)

# ✅ Use absolute path for model file
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

if __name__ == "__main__":
    app.run(debug=True)