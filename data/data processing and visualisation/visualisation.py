import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data():
    df = pd.read_csv('dataset.csv')
    
    os.makedirs('../images', exist_ok=True)

    # Plot target distribution
    plt.figure(figsize=(10,6))
    sns.countplot(x='NObeyesdad', data=df, palette='viridis')  # Target column
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

if __name__ == "__main__":
    visualize_data()
    print("Visualizations saved to /images folder")