def preprocess_data(df):
    # Handle categorical variables
    categorical_cols = ['strongest_subjects', 'preferred_task', 'future_career_goal', 'preferred_work_type', 'preferred_thinking_style']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    numerical_cols = ['programming_skills', 'interest_in_technology']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Preprocess the data
    processed_df = preprocess_data(df)

    return processed_df

def split_features_and_target(df, target_column):
    # Split the dataframe into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def save_preprocessed_data(df, output_file):
    # Save the preprocessed data to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Load and preprocess the data
    file_path = '../data/sample_students.csv'
    processed_data = load_and_preprocess_data(file_path)

    # Save the preprocessed data
    save_preprocessed_data(processed_data, '../data/preprocessed_students.csv')
