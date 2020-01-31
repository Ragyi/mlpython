# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# <h2>Model Appraisal in ML</h2>
# 
# <h3>Python Packages for Supervised Learning</h3>

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
#load CSV
data = pd.read_csv('/Users/ragyibrahim/Downloads/foodtruck_profits.txt', delimiter = ',', header = None).values
#Print data shape
print("Data Shape: {}".format(data.shape))
#Print Head
print("Data head: {}".format(data[0:5, :]))

#%% [markdown]
# Create Scatter Plot of Data

#%%
#create plot
plt.scatter(data[:, 0], data[:, 1], color = 'red', marker = 'x')
#Add x label
plt.xlabel("Population of city in 10,000s")
#Add y label
plt.ylabel("Profit of 10,000s")
plt.show()

#%% [markdown]
# Predict profit from Population

#%%
#create a variable for the length
m = len(data)

#Split label and feature variables into seperate matrices
X = np.matrix(data[:,0])
Y = np.matrix(data[:,1])

#The above creates a 1-D matrix. We need to transpose 
X = np.matrix(X).T
Y = np.matrix(Y).T

#Print results
print("Number of Examples: {}".format(m))
print("Shape of Data: {}".format(X.shape))
print("Shape of Label: {}".format(Y.shape))

#%% [markdown]
# <h3>Fitting a regression model</h3>
# 
# A regression line takes the form: $y = w_0 + w_1x_1$ <br>
# 
# The closed form solution is: $\text{W = }{(X^TX)}^{-1} X^Ty $<br>
# 
# Where $\text{ W = }[w_o, w_1] \text{ ,  X = }[1, x_1] $

#%%
#Add the intercept term to X
X = np.c_[np.ones(m), X]
print("New shape of data: {}".format(X.shape))

#%% [markdown]
# <code>np.linalg.pinv</code> calculates the inverse of avaliable and the pseudo inverse $A^+$ when determinant is zero and therefore the inverse is not possible. This is why it is preferred to <code>np.linalg.inv</code>

#%%
#Compute closed form solution
tempW = np.linalg.inv(np.dot(X.T, X))
tempW2 = np.dot(X.T, Y)
W = np.dot(tempW, tempW2)
print("Weights matrix shape: {}".format(W.shape))
print(W)

#%% [markdown]
# <h3>Plot regression line</h3>

#%%
#Plot data points
plt.scatter(data[:,0], data[:, 1], color = 'red', marker = "x")
#Plot regression line
plt.plot(X[:,1], np.dot(X,W))
#Add x and y labels
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in 10,000s')
#Show plot
plt.show()

#%% [markdown]
# <h3>Use RL to make predictions</h3>

#%%
#Add the intercept and the X-value
predicted_profit = np.dot([1,17.5], W)
# Predicted profit needs to be multiplied by y units, 10,000, and rounded to be currency
print ("The predicted Profit at x = 17.5: {}".format(np.round(predicted_profit*10000, decimals = 2)))
