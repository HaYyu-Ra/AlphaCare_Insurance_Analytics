import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load prepared data
df = pd.read_csv('data/prepared_data.csv')

# Example model training (replace with actual code)
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'model.pkl')
