import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_data(df):
    # Separate features and target
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_models(X_train, y_train, preprocessor):
    # Define models
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
    
    # Train and store all models
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'../images/{name.lower().replace(" ", "_")}_cm.png')
        plt.close()
    
    return results

def visualize_data(df):
    os.makedirs('../images', exist_ok=True)

    # Plot target distribution
    plt.figure(figsize=(10,6))
    sns.countplot(x='NObeyesdad', data=df, palette='viridis')
    plt.title('Obesity Level Distribution')
    plt.savefig('../images/target_distribution.png')
    plt.close()

    # Plot heatmap only for numerical features
    plt.figure(figsize=(12,8))
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.savefig('../images/correlation_heatmap.png')
    plt.close()

def main():
    # Load data
    df = pd.read_csv('dataset.csv')
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train models
    models = train_models(X_train, y_train, preprocessor)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    print("\nVisualizations and model evaluation saved to /images folder")

if __name__ == "__main__":
    main()