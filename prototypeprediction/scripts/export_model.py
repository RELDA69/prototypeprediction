from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

def export_model():
    # Load the dataset
    data = pd.read_csv('../data/sample_students.csv')

    # Preprocess the data (this should match the preprocessing done during training)
    # Assuming preprocess.py has a function called preprocess_data
    from preprocess import preprocess_data
    X, y = preprocess_data(data)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, 'ml/model.pkl')

if __name__ == "__main__":
    export_model()
