# Diabetes Dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. doad the dataset
df = pd.read_csv("diabetes.csv")   # replace path if needed
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# 2. data Pre-processing
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
print("\nStatistical Summary:\n", df.describe())

# 3. Handle Categorical Data
# In the Pima Indians Diabetes dataset, all columns are numeric,so no categorical encoding is needed.


# 4. universal analysis
# histograms for all numerical features
df.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Univariate Analysis - Histograms", fontsize=16)
plt.show()

# boxplots to check outliers
plt.figure(figsize=(12,6))
df.boxplot()
plt.title("Univariate Analysis - Boxplots")
plt.xticks(rotation=45)
plt.show()


# 5. bivariate analysis
# scatter plot: Glucose vs Insulin
plt.figure(figsize=(6,6))
plt.scatter(df["Glucose"], df["Insulin"], alpha=0.6, c=df["Outcome"], cmap="coolwarm")
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.title("Bivariate Analysis - Glucose vs Insulin")
plt.colorbar(label="Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.show()

# scatter plot: Age vs BMI
plt.figure(figsize=(6,6))
plt.scatter(df["Age"], df["BMI"], alpha=0.6, c=df["Outcome"], cmap="viridis")
plt.xlabel("Age")
plt.ylabel("BMI")
plt.title("Bivariate Analysis - Age vs BMI")
plt.colorbar(label="Outcome")
plt.show()

# compare Diabetes Outcome with Glucose levels using boxplot
plt.figure(figsize=(6,6))
df.boxplot(column="Glucose", by="Outcome")
plt.title("Bivariate Analysis - Glucose vs Outcome")
plt.suptitle("")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Glucose Level")
plt.show()

# compare Diabetes Outcome with Age using boxplot
plt.figure(figsize=(6,6))
df.boxplot(column="Age", by="Outcome")
plt.title("Bivariate Analysis - Age vs Outcome")
plt.suptitle("")
plt.xlabel("Outcome")
plt.ylabel("Age")
plt.show()