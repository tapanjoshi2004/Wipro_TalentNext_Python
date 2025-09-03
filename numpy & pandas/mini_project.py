import pandas as pd
import numpy as np

df = pd.read_csv("datasetExample.csv")

print("First 5 rows of dataset:")
print(df.head())

def detect_outliers_iqr(data):
    outlier_indices = {}
    for col in data.select_dtypes(include=[np.number]).columns: 
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        outlier_indices[col] = outliers.values
    return outlier_indices

outliers = detect_outliers_iqr(df)

print("\nDetected Outliers (by column):")
for col, values in outliers.items():
    if len(values) > 0:
        print(f"{col}: {values}")
    else:
        print(f"{col}: No outliers found")