# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>

#%% [markdown]
# <h2>Model Assessment</h2>
# 
# <h4>Confusion Matrix</h4>
#%% [markdown]
# ![](confusion.png)
#%% [markdown]
# ![](assesMetrics.png)
#%% [markdown]
# <h4>ROC Curve</h4>
#%% [markdown]
# Receiver Operating Characteristics (ROC) are used to depict the **trade off** between the true positive *rate* and false positive *rate*. ROC curve is especially useful for domains with imbalanced class distribution and unequal classification error costs.<br>
# <br>
# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. This has to be done to depict relative trade-offs between benefits (true positives) and costs (false positives).<br>
# <br>
# As you can see in the figure below, different methods can work better in different parts of ROC space. Lets say there are two algorithms like Alg 1 and Alg 2 in the figure. The Alg 2 is good in the sense that it can give you high true positive rate while keeping the false positive rate low. whereas in Alg 1, if it is allowed to incur more false positive rate, then Alg 1 can give us better higher true positive rate too.
#%% [markdown]
# ![](ROC.png)
#%% [markdown]
# Obviously it depends on the specification of the problem. how much can we afford false positive rate. if we can afford higher false positive rate, we can have higher true positive rate too.
# 
# A model that predicts at chance (random guessing) will have an ROC curve that looks like the diagonal dashed line in the figure above. That is not a discriminating model. The further the curve is from the diagonal line, the better the model is at discriminating between positives and negatives in general.
#%% [markdown]
# ![](diffROC.png)
#%% [markdown]
# There are useful statistics that can be calculated via ROC curve, like the Area Under the Curve (AUC) and the Youden Index. These tell you how well the model predicts and the optimal cut point for any given model (under specific circumstances). AUC is used to summarize the ROC curve using a single number. The higher the value of AUC, better performing is the classifier! A random classifier has an AUC of 0.5.
#%% [markdown]
# <h4>F-1 Measure</h4>
#%% [markdown]
# $F_1 = 2\times \frac{\text{Precision } \times \text{ Recall}}{\text{Precision + Recall}}$
#%% [markdown]
# <h3>Regression Metrics</h3>
# 
# <h4>Mean Square Error</h4>
# To measure how close the predictions are to the true target values, Mean Square Error (MSE) is a popular measure. MSE is defined as:
# 
# MSE = $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
# 
# Derived from MSE is Root Mean Square Error (RMSE):
# 
# RMSE = $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
# 
# and Mean Absolute Error (MAE):
# 
# MAE = $\frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$
# 
# 
# Statistically, the goal of any model should be to reduce the MSE because a smaller MSE implies that there is relatively little difference between the estimated and observed outputs. Generally speaking, a well-fitted model should have a relatively low MSE value. The ideal form has an MSE of zero, since it indicates that there is no difference between the estimated and observed parameters.

#%%
#Load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#%%
#Implement RMSE in python using SKlearn
from math import *
RMSE = sqrt(mean_squared_error(Ytest, yPredict))
print("Root mean squared error (RMSE):${}".format(RMSE))
