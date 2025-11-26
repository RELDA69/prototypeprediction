import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib

# Generate dummy dataset (100 samples)
np.random.seed(42)
data = {
    'strongest_subjects': [np.random.choice(['Programming', 'Mathematics', 'Web development / design', 'Networking / hardware', 'Data analysis', 'Business or management'], size=np.random.randint(1, 3), replace=False).tolist() for _ in range(100)],
    'task_enjoy': np.random.choice(['Designing websites or interfaces', 'Solving logic or programming problems', 'Fixing computers or networks', 'Analyzing data or organizing information', 'Creating games or graphics', 'Planning business or technical solutions'], 100),
    'prog_skills': np.random.randint(1, 6, 100),
    'tech_interest': np.random.randint(1, 6, 100),
    'career_want': np.random.choice(['Web/Mobile Developer', 'Software Engineer', 'Network Administrator / Cybersecurity', 'Data Analyst / Database Admin', 'Game Developer / Graphics', 'Business Analyst / Information Systems'], 100),
    'prefer_work': np.random.choice(['Design', 'Code', 'Hardware'], 100),
    'prefer_creative_logical': np.random.choice(['Creative', 'Logical', 'Both'], 100),
    'major': np.random.choice(['Computer Science', 'Information Technology', 'Software Engineering', 'Data Science', 'Cybersecurity', 'Business Information Systems'], 100)  # Target
}
df = pd.DataFrame(data)

# Preprocessing
mlb = MultiLabelBinarizer()
subjects_encoded = pd.DataFrame(mlb.fit_transform(df['strongest_subjects']), columns=mlb.classes_, index=df.index)

le_task = LabelEncoder()
le_career = LabelEncoder()
le_work = LabelEncoder()
le_creative = LabelEncoder()
le_major = LabelEncoder()

df['task_enjoy_encoded'] = le_task.fit_transform(df['task_enjoy'])
df['career_want_encoded'] = le_career.fit_transform(df['career_want'])
df['prefer_work_encoded'] = le_work.fit_transform(df['prefer_work'])
df['prefer_creative_logical_encoded'] = le_creative.fit_transform(df['prefer_creative_logical'])
df['major_encoded'] = le_major.fit_transform(df['major'])

# Combine features
X = pd.concat([subjects_encoded, df[['task_enjoy_encoded', 'prog_skills', 'tech_interest', 'career_want_encoded', 'prefer_work_encoded', 'prefer_creative_logical_encoded']]], axis=1)
y = df['major_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(mlb, 'mlb.pkl')
joblib.dump(le_task, 'le_task.pkl')
joblib.dump(le_career, 'le_career.pkl')
joblib.dump(le_work, 'le_work.pkl')
joblib.dump(le_creative, 'le_creative.pkl')
joblib.dump(le_major, 'le_major.pkl')

print("Model trained and saved.")

# Function to preprocess new input
def preprocess_input(data):
    mlb = joblib.load('mlb.pkl')
    le_task = joblib.load('le_task.pkl')
    le_career = joblib.load('le_career.pkl')
    le_work = joblib.load('le_work.pkl')
    le_creative = joblib.load('le_creative.pkl')
    
    subjects = pd.DataFrame(mlb.transform([data['strongest_subjects']]), columns=mlb.classes_)
    task_encoded = le_task.transform([data['task_enjoy']])[0]
    career_encoded = le_career.transform([data['career_want']])[0]
    work_encoded = le_work.transform([data['prefer_work']])[0]
    creative_encoded = le_creative.transform([data['prefer_creative_logical']])[0]
    
    processed = subjects.values.flatten().tolist() + [task_encoded, int(data['prog_skills']), int(data['tech_interest']), career_encoded, work_encoded, creative_encoded]
    return processed