# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>

#%% [markdown]
# <h2>Multivariate Regression</h2>

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
#import CSV function
def importCSV(path,delim,headers):
   globals() ['dataFrame'] = pd.read_csv(path, delimiter=delim, header=headers).values
   print("Data Shape: {}".format(dataFrame.shape))
   print("Data head: {}".format(dataFrame[0:5, :]))

#%% [markdown]
# <h4>Import data</h4>

#%%
myDataFrame = pd.read_csv('/Users/ragyibrahim/Downloads/advertising.csv')
featureCols = ['TV', 'Radio', 'Newspaper']
label = ['Sales']


#%%
X = myDataFrame[featureCols].values
Y = myDataFrame[label].values

print("feature columns: {}".format(X.shape))
print("label column: {}".format(Y.shape))

#%% [markdown]
# We can now build our regression model. We proceed by dividing this original dataset into two parts:
# 
# Training dataset: containing  of original data
# Testing dataset: containing  of original data
# Why do we do this? So that we can effectively judge our model. We build our model using training data. Then we test its performance on the unseen(by the model) test data.
# 
# For now, lets manually create the train/test split using  training data and  testing data.

#%%
#create training datasets
Xtrain = X[:140]
Ytrain = Y[:140]
#Create testing datasets
Xtest = X[140:]
Ytest = Y[140:]
#Print shapes
print("Feature Training dataset{}".format(Xtrain.shape), "Feature Testing dataset{}".format(Xtest.shape))
print("Label Training dataset{}".format(Ytrain.shape), "Label Testing dataset{}".format(Ytest.shape))

#%% [markdown]
# <h4>Linear regression model from sci-kit learn</h4>

#%%
from sklearn.linear_model import LinearRegression


#%%
#Build model parameters
myLRModel = LinearRegression()
#For model to training data
myLRModel.fit(Xtrain, Ytrain)

#%% [markdown]
# Now that we have built our model, lets see how it performs with our test dataset

#%%
#make predictions
yPredict = myLRModel.predict(Xtest)

#Print results
print("Predicted sales: {}".format(yPredict))
#Print Ground truth
print("True sales: {}".format(Ytest))
