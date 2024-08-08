import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest


# Load the data
data = pd.read_csv(r'C:\Users\Qazafi\Desktop\AmesHousing.csv')

# Data Cleaning and processing
print(data.head())
print(data.isnull().sum())

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Fill missing values for numeric columns only
data.fillna(data.mean(), inplace=True)

# Define Features and Target variable
X = data.drop('SalePrice', axis=1)
Y = data['SalePrice']

# Normalize features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Normalize target variable using StandardScaler
scaler_Y = StandardScaler()
Y = Y.values.reshape(-1, 1)
Y_scaled = scaler_Y.fit_transform(Y).ravel()


# Use Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination as needed
outliers = iso_forest.fit_predict(X)
outliers = outliers == -1  # Outliers are labeled as -1

# Draw the Graph to see the Outliers in Data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 7], c=outliers, cmap='coolwarm', alpha=0.7)
plt.xlabel('Order')
plt.ylabel('Alley')
plt.title('Isolation Forest Outliers')
plt.colorbar(label='Outlier')
plt.show()

# IN Graph, the Red color show the outlier in the data
# The above graph is only showing the graph of feature 1 and feature 7
# Remove outliers
X_cleaned = X[~outliers]
Y_cleaned = Y_scaled[~outliers]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_cleaned, Y_cleaned, test_size=0.2, random_state=42)

# Define the model
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42 , eta0=0.001)

# Train the model
sgd_regressor.fit(X_train, Y_train)

# Make predictions on the testing data
Y_pred = sgd_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
