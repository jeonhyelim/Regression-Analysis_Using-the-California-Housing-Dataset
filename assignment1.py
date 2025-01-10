#!/usr/bin/env python
# coding: utf-8

# In[11]:


pip install scikit-learn


# In[12]:


pip install scipy


# In[13]:


# Step 1: Load the California Housing Dataset
#import necessary libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd

#load the California Housing dataset
california = fetch_california_housing()

#split the data into features (X) and target(y)
X = pd.DataFrame(california.data, columns=california.feature_names) #feature variables
y = pd.Series(california.target) #Target variable(house prices)

#Check the shape of the data
print("Feature data shape:", X.shape)
print("Target data shape:", y.shape)

#Display the first few rows of the dataset
print(X.head())
print(y.head())


# In[14]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Load the California Housing Dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target)

# Use 'MedInc' (Median Income) as the feature for single-variable regression
X = X[['MedInc']]

# ---------- Task 1: Regression using Least Squares ----------
# Step 2: Add a bias (intercept) term to the features
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Step 3: Calculate theta using the normal equation
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
intercept_task1, slope_task1 = theta_best
y_pred_task1 = np.dot(X_b, theta_best)
mse_task1 = mean_squared_error(y, y_pred_task1)

print("Task 1 - Optimal parameters:")
print(f"Intercept(theta_0): {intercept_task1}")
print(f"Slope(theta_1): {slope_task1}")
print(f"Mean Squared Error(MSE): {mse_task1}")
print("")


# ---------- Task 2: Regression using Maximum Likelihood Estimation (MLE) ----------
# Define the negative log-likelihood function
def negative_log_likelihood(params, X, y):
    intercept, slope, sigma_sq = params
    n = len(y)
    y_pred = intercept + slope * X[:, 1]
    nll = (n / 2) * np.log(2 * np.pi * sigma_sq) + (1 / (2 * sigma_sq)) * np.sum((y - y_pred) ** 2)
    return nll

# Initial parameters for optimization
initial_params = [0, 0, 1]
result = minimize(negative_log_likelihood, initial_params, args=(X_b, y), method='L-BFGS-B', bounds=[(-10, 10), (-10, 10), (1e-5, None)])
theta_0, theta_1, sigma_sq_mle = result.x
y_pred_task2 = theta_0 + theta_1 * X_b[:, 1]
mse_task2 = mean_squared_error(y, y_pred_task2)

print("Task 2 - Optimal parameters(MLE):")
print(f"Intercept(theta_0): {theta_0}")
print(f"Slope(theta_1): {theta_1}")
print(f"Variance(sigma^2): {sigma_sq_mle}")
print(f"Mean Squared Error(MSE): {mse_task2}")
print("")

# ---------- Task 3: Regression using Scikit-learn ----------
# Initialize and fit the Linear Regression model using Scikit-learn
model = LinearRegression()
model.fit(X, y)
intercept_sklearn = model.intercept_
slope_sklearn = model.coef_[0]
y_pred_task3 = model.predict(X)
mse_task3 = mean_squared_error(y, y_pred_task3)

print("Task 3 - Optimal parameters(Scikit-learn):")
print(f"Intercept(theta_0): {intercept_sklearn}")
print(f"Slope(theta_1): {slope_sklearn}")
print(f"Mean Squared Error(MSE): {mse_task3}")

# ---------- Visualization: Actual vs Predicted values for all tasks ----------
plt.figure(figsize=(10, 6))

# Scatter plot for actual values
plt.scatter(X, y, color='blue', label='Actual House Prices')

# Plot Task 1 prediction line (Least Squares)
plt.plot(X, y_pred_task1, color='green', linestyle='--', label='Task 1: Least Squares')

# Plot Task 2 prediction line (MLE)
plt.plot(X, y_pred_task2, color='orange', linestyle='-.', label='Task 2: MLE')

# Plot Task 3 prediction line (Scikit-learn)
plt.plot(X, y_pred_task3, color='red', linestyle='dotted', label='Task 3: Scikit-learn')

# Add labels, title, and legend
plt.xlabel('Median Income(MedInc)')
plt.ylabel('House Prices')
plt.title('Regression Comparison: Task 1(Least Squares), Task 2(MLE), and Task 3(Scikit-learn)')
plt.legend()

# Show the plot
plt.show()


# In[ ]:




