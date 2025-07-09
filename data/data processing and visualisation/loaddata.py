import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')

print(df.head())
print(df.info())
print(df.describe())

sns.countplot(x='NObeyesdad', data=df)
plt.title('Class Distribution')
plt.show()
