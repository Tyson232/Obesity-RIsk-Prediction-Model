import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# Load dataset
df = pd.read_csv('dataset.csv')  # Ensure this file exists in the same folder

# Separate features and target
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Encode all categorical features in X
feature_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

# Encode the target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Create output folder
os.makedirs("model", exist_ok=True)

# Save model and encoders
joblib.dump(model, 'model/gb_model.pkl')
joblib.dump(feature_encoders, 'model/gender_encoder.pkl')
joblib.dump(target_encoder, 'model/target_encoder.pkl')

print("✅ Model and encoders saved successfully.")
