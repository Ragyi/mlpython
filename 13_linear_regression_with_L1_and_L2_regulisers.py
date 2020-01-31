# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# <h2>Linear Regression with $L_1$ and $L_2$ Regulisers</h2>

#%%
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
#load data into pandas DF
data = pd.read_csv('/Users/ragyibrahim/Downloads/poly_data.csv')
#Number of rows and cols
rows, cols = data.shape
print("Data has {} rows and {} columns".format(rows, cols))
#Preview data
data.head()


#%%
#Isolate predictors and response variable
predictors = ['x']
response = ['y']
#Plot data 
plt.plot(data[predictors], data[response], '.')
plt.xlabel('Predictors')
plt.ylabel('Response')
plt.title('Viz our data')
plt.show()


#%%
#Fit model to data
from sklearn.linear_model import LinearRegression

lm = LinearRegression(normalize=True)
lm_fit = lm.fit(data[predictors], data[response])
yPredict = lm.predict(data[predictors])


#%%
#Evaluate the model
mse_1 = np.mean((yPredict - data[response])**2)
print("Model MSE: {}".format(mse_1[0]))
plt.plot(data['x'], data['y'], '.', data['x'], yPredict, '-')
plt.show()

#%% [markdown]
# <h3>Linear regression with polynomial features (polynomial regression)</h3>
# 
# For linear regression, our hypothesis function was in the form $h(x)=wx+b$.
# 
# We have to use polynomial regression to get a better fit. For this, we have to generate extra features as powers of feature . So our final model will be of the form $h(x)=w_{n}x^{n}+w_{n-1}x^{n-1}+\cdots+w_{2}x^{2}+wx+b$
# We can use the same Linear Regression from Scikit learn for this, but we now have to generate the features as powers of $x$.

#%%
#We are going to add columns to our exiting data frame

#To generate a name starting with a character and ending with a number, lets try this:
print("x_%d"%5)

#%% [markdown]
# Here, “%d” (as found in coding languages C, C++, Matlab) stands for digit, and is replaced by the value following % after the quotes. We can now generate column names and columns in our dataframe as:

#%%
#take numbers in x to power 2,3,4 and 5 to generate the new features
for i in range(2,6):
    colname = 'x_%d'%i
    data[colname] = data.x**i
data.head()


#%%
#Fit new model
newPredictors = data.columns.values[1:]
lm_2 = LinearRegression(normalize= True)
lm_2.fit(data[newPredictors], data[response])
yPredict_2 = lm_2.predict(data[newPredictors])
#Evaluate model
mse_2 = np.mean((yPredict_2 - data[response])**2)
print("LM Model 2 MSE: {}".format(mse_2[0]))
plt.plot(data['x'], data['y'], '.', data['x'], yPredict_2, '-')

#%% [markdown]
# model with features until $x^{15}$

#%%
#generate dataset
for i in range(2,16):
    colname = 'x_%d'%i
    data[colname] = data.x**i
data.head()


#%%
#Fot the polynomial model
polyPredictors = data.columns.values[1:]
lm_3 = LinearRegression(normalize= True)
lm_3.fit(data[polyPredictors], data[response]) #<=Minimising the sum of squares
yPredict_3 = lm_3.predict(data[polyPredictors])

#Visualise model fit to training data
plt.plot(data['x'], data['y'], '.', data['x'], yPredict_3, '-')
mse_3 = np.mean((data[response] - yPredict_3)**2)
print('Model MSE: {}'.format(mse_3[0]))

#%% [markdown]
# Lets check the MSE of each of our models. Lets also check the value of our the coefficients:

#%%
print("MSE Simple LR: {}".format(mse_1[0]))
print("MSE Polynomial LR with power = 3:  {}".format(mse_2[0]))
print("MSE Polynomial LR with power = 15: {}".format(mse_3[0]))

#%% [markdown]
# <h3>$L_2$ Regularised linear regression</h3>
# 
# $\min_{w}\ \frac{1}{n} \sum_{i}^{} \ L(y_i,\text{x}_i^T\text{w}) + \lambda_2 ||\text{w}||^2_2$

#%%
#Load Ridge Reguliser module
from sklearn.linear_model import Ridge

#Call the ridge regression model
ridgeLm = Ridge(alpha = 0.003, normalize = True) #Here alpha is equivilant to lamda
#fit the data to the model
ridgeLm.fit(data[polyPredictors], data[response])
#Generate predictions
yPredictRidge_1 = ridgeLm.predict(data[polyPredictors])

#Evaluate and Visualise
ridgeMSE = np.mean((yPredictRidge_1 - data[response])**2)
print("Ridge Model MSE: {}".format(ridgeMSE[0]))
##Visualise
plt.plot(data['x'], data['y'], '.', data['x'], yPredictRidge_1, '-')
plt.show()

#%% [markdown]
# <h3>$L_1$ Regularised linear regression</h3>
# 
# $\min_{w}\ \frac{1}{n} \sum_{i}^{} \ L(y_i,\text{x}_i^T\text{w}) + \lambda_1 ||\text{w}||_1$
# 
# $L_1$ regularisation or LASSO is another regulariser which is of the form:

#%%
#import module
from sklearn.linear_model import Lasso
#Call the lasso regression model with penalty (alpha) = 0.01
#We also specify the max number of iterations as 10^
lassReg = Lasso(alpha = 1e-3, normalize = True, max_iter = 1e5)
lassReg.fit(data[polyPredictors], data[response])
yPredictLasso = lassReg.predict(data[polyPredictors])
#Evaluate and Visualise
lassoMSE = np.mean((yPredictLasso - data['y'])**2)
print("LASSO Model MSE: {}".format(lassoMSE))
##Visualise
plt.plot(data['x'], data['y'], '.', data['x'], yPredictLasso, '-')
plt.show()
print("Model Coeff: {}".format(lassReg.coef_))
