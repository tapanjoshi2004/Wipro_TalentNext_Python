import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("melb_data.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 2.Handle inappropriate data
df.drop_duplicates(inplace=True)        # if any duplicate row or coloum present in this dataset, remove or drop it.

if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

# 3. Handle missing data
print("\nMissing values before handling:")
print(df.isnull().sum())

# option 1
df.dropna(axis=1, thresh=len(df)*0.5, inplace=True) 

# option 2
for col in df.columns:
    if df[col].dtype == "object":  
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:                          
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# 4. Handle categorical data
categorical_cols = df.select_dtypes(include=["object"]).columns         # convert categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataFrame after encoding categorical variables:")
print(df.head())

# final dataset
print("\nFinal Shape of DataFrame:", df.shape)