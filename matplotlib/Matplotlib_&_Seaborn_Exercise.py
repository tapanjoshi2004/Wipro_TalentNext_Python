# 1 .......................................................

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')

print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()

sns.countplot(data=df, x='Gender')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.show()


# 2 ..........................................................

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('salary_data.csv')

print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.histplot(df['salary'], kde=True)
plt.show()

if 'experience' in df.columns:
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='experience', y='salary')
    plt.show()


# 3 ..................................................................

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Social_Network_Ads.csv')

print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.countplot(data=df, x='Purchased')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Age', y='EstimatedSalary', hue='Purchased')
plt.show()

