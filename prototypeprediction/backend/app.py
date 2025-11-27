from flask import Flask, request, jsonify
from flask_cors import CORS   # 👈 add this
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)   # 👈 enable CORS for all routes

# Load trained pipeline
MODEL_PATH = Path(__file__).parent / "models" / "model_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        input_df = pd.DataFrame([{
            "Year": data.get("Year"),
            "Technical Skills": data.get("Technical Skills"),
            "Programming Languages": data.get("Programming Languages"),
            "Programming Languages Ratings": data.get("Programming Languages Ratings"),
            "Soft Skills": data.get("Soft Skills"),
            "Soft Skills Rating": data.get("Soft Skills Rating"),
            "Projects": data.get("Projects"),
            "Career Interest": data.get("Career Interest"),
            "Challenges": data.get("Challenges"),
            "Support required": data.get("Support required"),
            "Method": data.get("Method")
        }])

        prediction = pipeline.predict(input_df)[0]
        return jsonify({"predicted_major": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)