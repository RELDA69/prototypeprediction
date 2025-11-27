from flask import Flask
from flask_cors import CORS
from pathlib import Path
import joblib
from api.routes import create_api_blueprint

app = Flask(__name__)
CORS(app)

# Load model once here
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# Pass model into blueprint
api = create_api_blueprint(model)
app.register_blueprint(api)

if __name__ == "__main__":
    app.run(debug=True)