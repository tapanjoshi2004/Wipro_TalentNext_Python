# 1. Sales Prediction ...........................................................................................................
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the data in the dataframe
df = pd.read_csv('Advertising.csv')

# perform data reprocessing
print(df.head())
print(df.describe())
print(df.isnull().sum())  

# Handle Categorical Data -- (not needed here as data is numerical)


# Perform Exploratory Data Analysis
sns.pairplot(df)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Build the model using Multiple Linear Regression
X = df[['TV', 'Radio', 'Newspaper']] 
y = df['Sales'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Use the appropriate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)



# 2. Diabetes prediction .....................................................................................................
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the PIMA Indians diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
df = pd.read_csv(url, names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

# perform data preprocessing
print(df.head())
print(df.describe())
print(df.isnull().sum())  


X = df.drop('Outcome', axis=1)  
y = df['Outcome']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform Exploratory Data Analysis
sns.pairplot(df, hue='Outcome')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Build the model using Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)

# Use the appropriate evaluation metrics
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(classification_report(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))

# Build the model using K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn)}")
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))