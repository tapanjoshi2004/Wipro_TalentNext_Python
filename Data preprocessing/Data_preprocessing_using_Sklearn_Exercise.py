# 1 .......................................................

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("melb_data.csv")

print("First 5 rows:\n", df.head())
print("\nShape of dataset:", df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values before handling:\n", df.isnull().sum())

df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = num_imputer.fit_transform(df[[col]])
    else:
        df[col] = cat_imputer.fit_transform(df[[col]])

print("\nMissing Values after handling:\n", df.isnull().sum())

categorical_cols = df.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder(drop="first", sparse=False)
encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))

encoded.columns = encoder.get_feature_names_out(categorical_cols)
df = pd.concat([df.drop(categorical_cols, axis=1), encoded], axis=1)

print("\nData after encoding categorical variables:\n", df.head())

if "Price" in df.columns:
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Price"] >= Q1 - 1.5*IQR) & (df["Price"] <= Q3 + 1.5*IQR)]
    print("\nShape after removing outliers:", df.shape)

scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nFinal Preprocessed Data (first 5 rows):\n", df.head())
print("\nFinal Shape of Dataset:", df.shape)