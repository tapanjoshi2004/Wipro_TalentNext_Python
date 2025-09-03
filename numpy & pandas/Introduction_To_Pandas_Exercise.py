# 1.....................................................................

import pandas as pd

cars = pd.read_csv('cars.csv')

print("First 10 rows:")
print(cars.head(10))

print("\nDataFrame cars:")
print(cars)

print("\nLast 5 rows:")
print(cars.tail(5))

print("\nMeta information:")
print(cars.info())
print("\nDescription:")
print(cars.describe())



# 2...........................................................................

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

startups = pd.read_csv('50_startups.csv')

print("Statistical summary:")
print(startups.describe())

print("\nCorrelation coefficient with Profit:")
print(startups.corr()['Profit'].sort_values(ascending=False))

plt.figure(figsize=(10,8))
sns.heatmap(startups.corr(), annot=True, cmap='coolwarm')
plt.show()