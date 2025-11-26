from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

class Recommender:
    def __init__(self, model_path='ml/model.pkl'):
        self.model = joblib.load(model_path)
        self.features = ['strongest_subjects', 'preferred_task', 'programming_skills', 
                         'interest_in_technology', 'future_career_goal', 
                         'preferred_work_type', 'preferred_thinking_style']

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure the input data has the same features as the model
        input_df = input_df[self.features]
        
        # Make prediction
        prediction = self.model.predict(input_df)
        return prediction[0]  # Return the predicted major

    def get_recommendations(self, input_data):
        predicted_major = self.predict(input_data)
        # Here you can add logic to provide additional recommendations based on the predicted major
        return predicted_major
