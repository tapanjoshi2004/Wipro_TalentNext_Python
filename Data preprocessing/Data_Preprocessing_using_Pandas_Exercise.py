# 2 ......................................................

import pandas as pd
import numpy as np

df = pd.read_csv("melb_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values (before handling):")
print(df.isnull().sum())

df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)

for col in df.columns:
    if df[col].dtype == "object":  
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:                          
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values (after handling):")
print(df.isnull().sum())

df.drop_duplicates(inplace=True) 

if 'Address' in df.columns:                   
    df.drop(columns=['Address'], inplace=True)

categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nData after encoding categorical variables:")
print(df.head())

print("\nCorrelation Matrix (Top 10 features with Price):")
if "Price" in df.columns:
    corr_matrix = df.corr(numeric_only=True)["Price"].sort_values(ascending=False).head(10)
    print(corr_matrix)

print("\nFinal Shape of DataFrame:", df.shape)