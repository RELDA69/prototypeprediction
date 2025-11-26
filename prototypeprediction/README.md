# prototypeprediction
## CS/IT Major Recommender

This project is a full-stack web application that predicts a student's recommended IT/CS major based on their interests, skills, and career goals. It consists of a Flask backend that serves a machine learning model and a frontend built with HTML, CSS, and JavaScript.

### Project Structure
```
prototypeprediction
├── backend
│   ├── app.py                # Flask application setup
│   ├── api
│   │   └── routes.py         # API routes for predictions
│   ├── models
│   │   └── recommender.py     # Model logic for recommendations
│   ├── ml
│   │   ├── train.py          # Model training script
│   │   ├── preprocess.py      # Data preprocessing script
│   │   └── model.pkl          # Trained machine learning model
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Docker configuration for deployment
├── frontend
│   ├── templates
│   │   └── home.html         # Main HTML page for user input
│   └── static
│       ├── css
│       │   └── styles.css     # Styles for the frontend
│       └── js
│           └── app.js        # JavaScript for handling user input and API calls
├── notebooks
│   └── model_experiment.ipynb # Jupyter notebook for model experimentation
├── data
│   └── sample_students.csv    # Sample dataset for training the model
├── tests
│   ├── test_api.py           # Tests for API endpoints
│   └── test_model.py         # Tests for model predictions
├── scripts
│   └── export_model.py       # Script for exporting the trained model
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

### Setup Instructions

1. **Backend Setup**
   - Navigate to the `backend` directory:
     ```
     cd backend
     ```
   - Set up a virtual environment:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Run the Flask app:
     ```
     python app.py
     ```

2. **Frontend Setup**
   - Ensure the frontend is served using a simple HTTP server or a framework.

### Dataset Preparation and Model Training

1. Store the CSV file in `data/sample_students.csv`.
2. The dataset must contain the following columns:
   - `strongest_subjects` (categorical)
   - `preferred_task` (categorical)
   - `programming_skills` (integer, 1-5)
   - `interest_in_technology` (integer, 1-5)
   - `future_career_goal` (categorical)
   - `preferred_work_type` (categorical)
   - `preferred_thinking_style` (categorical)

3. Preprocess and encode values using the `preprocess.py` script.
4. To retrain the model when the dataset is updated, run the `train.py` script after updating the CSV.
5. The backend loads the model from `ml/model.pkl` during startup.

### Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

### License
This project is licensed under the MIT License. See the LICENSE file for details.