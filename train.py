import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create directory for models if it doesn't exist
if not os.path.exists('gb_models'):
    os.makedirs('gb_models')

# Load the dataset
df = pd.read_csv('dataset.csv')

# Separate target variable
target = 'NObeyesdad'
y = df[target]
X = df.drop(columns=[target, 'id'])  # Drop ID as it's not a feature

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
joblib.dump(le_target, 'gb_models/target_encoder.pkl')

# Train main model (predicting NObeyesdad)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                         max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)

# Evaluate
y_pred = gb_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Main model accuracy: {acc:.4f}")

# Save main model
joblib.dump(gb_classifier, 'gb_models/main_gb_model.pkl')

# Now train models for each feature that can be predicted
features_to_model = [col for col in X.columns if col not in ['id']]

for feature in features_to_model:
    try:
        # Prepare data for this feature
        temp_X = X.drop(columns=[feature])
        temp_y = X[feature]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, test_size=0.2, random_state=42)
        
        # Determine if classification or regression
        if temp_y.nunique() < 20:  # Arbitrary threshold for classification
            model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, 
                                            max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(f"Model for {feature} (classification) - Accuracy: {score:.4f}")
        else:
            model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, 
                                            max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred, squared=False)
            print(f"Model for {feature} (regression) - RMSE: {score:.4f}")
        
        # Save model
        joblib.dump(model, f'gb_models/gb_model_for_{feature}.pkl')
        
    except Exception as e:
        print(f"Error modeling {feature}: {str(e)}")

# Save label encoders
joblib.dump(label_encoders, 'gb_models/label_encoders.pkl')

print("All models saved to gb_models directory")