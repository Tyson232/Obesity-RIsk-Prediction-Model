import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

class BlendingClassifier:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        # Train base models
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        # Get base model predictions
        meta_features = np.column_stack([
            model.predict_proba(X) for name, model in self.base_models.items()
        ])
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
    def predict(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X) for name, model in self.base_models.items()
        ])
        return self.meta_model.predict(meta_features)

def preprocess_data(df):
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_base_models(X_train, y_train, preprocessor):
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'KNN': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier())
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        plt.figure(figsize=(8,6))
        sns.heatmap(results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{name.lower().replace(" ", "_")}_cm.png')
        plt.close()
    
    return results

def implement_ensemble_techniques(X_train, X_test, y_train, y_test, preprocessor, base_models):
    # 1. Bagging
    bagging_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 2. Boosting
    boosting_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    # 3. Voting
    voting_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', VotingClassifier(
            estimators=[(name, model.named_steps['classifier']) for name, model in base_models.items()],
            voting='soft'
        ))
    ])
    
    # 4. Stacking
    stacking_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', StackingClassifier(
            estimators=[(name, model.named_steps['classifier']) for name, model in base_models.items()],
            final_estimator=LogisticRegression(max_iter=1000)
        ))
    ])
    
    # 5. Blending
    blended_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', BlendingClassifier(
            base_models={name: model.named_steps['classifier'] for name, model in base_models.items()},
            meta_model=LogisticRegression(max_iter=1000)
        ))
    ])
    
    ensembles = {
        'Random Forest': bagging_model,
        'Gradient Boosting': boosting_model,
        'Voting Classifier': voting_model,
        'Stacking Classifier': stacking_model,
        'Blending Classifier': blended_model
    }
    
    results = {}
    for name, model in ensembles.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }
    
    return results

def main():
    # Load your obesity dataset
    df = pd.read_csv('dataset.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train base models
    base_models = train_base_models(X_train, y_train, preprocessor)
    
    # Evaluate base models
    base_results = evaluate_models(base_models, X_test, y_test)
    
    # Test ensemble methods
    ensemble_results = implement_ensemble_techniques(
        X_train, X_test, y_train, y_test, preprocessor, base_models
    )
    
    # Print results
    print("\n=== Base Models ===")
    for name, res in base_results.items():
        print(f"{name}: Accuracy = {res['accuracy']:.4f}")
    
    print("\n=== Ensemble Methods ===")
    for name, res in ensemble_results.items():
        print(f"{name}: Accuracy = {res['accuracy']:.4f}")

if __name__ == "__main__":
    main()