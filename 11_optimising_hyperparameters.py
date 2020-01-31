# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>

#%% [markdown]
# <h2>Optimising Model Hyperprameters</h2>
# 
# In machine learning, a hyperparameter is a parameter whose value is set before the learning process begins. This means the value of a hyperparameter in a model cannot be estimated from data. They are often used in processes to help estimate model parameters.<br>
# Hyperparameters can often be set using heuristics Often they are tuned for a given predictive modelling problem. To search for the best hyperparameters, we need to partition training data into separate training and validation sets.
#%% [markdown]
# <h4>Validation Dataset</h4>
# 
# A validation set is a sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters.
# 
# The validation set is used to evaluate a given model and also to fine-tune the model hyperparameters. So, given a choice of hyperparameter values, you use the training set to train the model. But, how do you set the values for the hyperparameters? That’s what the validation set is for. You can use it to evaluate the performance of your model for different combinations of hyperparameter values (e.g. by means of a grid search process) and keep the best trained model.
# 
# 
#%% [markdown]
# But, how can we find the best hyperparameter?
# 
# - First, we need to decide on a possible range for hyperparameters. For example, a bounded interval such as $[0,1]$
# - We then define a search grid within the specified range. For example, we might like to select these values: $[0,10^{-3}, 10^{-2}, 10^{-1}, 1]$ for as hyperparameters in order to evaluate the model with them.
# - Next, we train a model using each hyperparameter value from the search grid and assess its performance on a validation set (separated from the training set).
# - Finally, we compute the performance on the validation set for each hyperparameter value and select the one with the best performance. Once the model is working with the best hyperparameter we defined it’s ready to be tested on separate test data.
#%% [markdown]
# <h3>Internal cross-validation</h3>
# 
# All the techniques that we previously discussed for model assessment are applicable for training/validation set splitting:
# 
# Random subsampling
# Stratified subsampling
# Cross-validation
# We are still assessing how a particular hyperparameter is doing on the validation set. Remember, this step is internal to the learning process and different from model assessment on the test data.
# 
# Let us examine how an internal cross-validation works. Instead of using a single validation set, we can use cross-validation within a training set to select the best set of hyperparameters. So basically it is exactly the same as the one we saw for test/train partitioning. However, in here we partition the data into training/validation sets. The following figure illustrates this process.
#%% [markdown]
# ![](internalCV.png)
#%% [markdown]
# Lets work on another example of that.
# 
# - Say, we want to do -fold Cross-validation to estimate the model performance of Elastic Net model.
# - We can divide the data into  equal subsamples and then train the model using 9 subsamples and test the model using the subsample. We repeat this -times using each subsample for the test purpose and all other subsamples for the training.
# - In the above train the model step, best hyperparameter can be selected using an internal cross-validation. If we want to use -fold cross-validation for this. Then for each possible hyperparameter set, we compute -fold cross validation (CV) accuracy and select the best hyperparameter set.
# - So in this example, we have an external -fold cross-validation for partitioning training/testing. Also we ran a -fold cross-validation for partitioning training/validation inside the training set for finding the best hyperparameters.
# 
# Remember that we can select the best hyperparameter set by searching/or optimizing over all possible values. Let us show you 3 possible ways to navigate the hyperparameter space:
# 
# - Grid-search (not so efficient). This is what we are using and explaining!
# - Random search (efficient in certain scenarios) [Bergstra et al. (JMLR 2012)
# - [Bayesian Optimization for Hyperparameter Tuning](https://arimo.com/data-science/2016/bayesian-optimization-hyperparameter-tuning) [Snoek et al. (2012)]