import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Load the dataset
    data = pd.read_csv('data/sample_students.csv')

    # Preprocess the data
    X = data[['strongest_subjects', 'preferred_task', 'programming_skills', 
               'interest_in_technology', 'future_career_goal', 
               'preferred_work_type', 'preferred_thinking_style']]
    y = data['recommended_major']

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'ml/model.pkl')

if __name__ == "__main__":
    train_model()