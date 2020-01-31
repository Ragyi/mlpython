# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# <h3>The Sigmoid Function</h3>
# 
# It seems that we need another function such as a decision function $\sigma(h(\text{x}))$ which uses a fixed non-linear link function and projects the value of $h(\text{x})$ into interval.<br>
# <br>
# Logistic regression does not directly model $y$ in terms $x$. Instead, it models something called logit value or log of odds against $x$ via linear regression. So generally we are modelling log of odds based on $x$.
#%% [markdown]
# The odds of a class $-1$ is defined as:
# 
# $\text{Odds = } \frac{P(y=1\vert\textbf{x})}{1-P(y=1\vert\text{x})}$
# 
# The odds that a randomly chosen day of the week is a weekend are:
# 
# $\frac{\frac{2}{7}}{1-(\frac{2}{7})} = \frac{2}{5}$
# 
# This Log of odds is called logit. Logistic regression uses the following linear model:
# 
# $log\frac{P(y=1\vert\text{x})}{1-P(y=1\vert\text{x})} = \text{w}_0 + \text{w}_1 \text{x}$
# 
# We can estimate this probability using the following equation:
# 
# $P(y=1\vert\text{x}) = \frac{1}{1+exp{(-\text{x}^T\text{w})}}$
# 
# The model leads to the following classification rules:
# 
# - when $\text{x}^T\text{w} > 0, P(y=1\vert\text{x}) > 0.5$, we decide in favour of class 1
# 
# - when $\text{x}^T\text{w} < 0, P(y=1\vert\text{x}) < 0.5$, we decide in favour of class -1
# 
# - when $\text{x}^T\text{w} = 0, P(y=1\vert\text{x}) = 0.5$, both classes are equally possible
# 
# <h4>Testing the model</h4>
# 
# So assume you have trained a logistic regression model and you have come up with proper values of $\text{w}$. Now by having a test point such as $\text{x}$, you calculate the value of $\text{x}^T\text{w}$.
# 
# - If this value is $\text{x}^T\text{w} > 0$ then it means $P(y=1\vert\text{x}) > 0.5$, the point is allocated to class 1.
# - On the other hand, if the value of $\text{x}^T\text{w} < 0$ then it means $P(y=1\vert\text{x}) < 0.5$, the point is allocated to class 0.
# - In case of $\text{x}^T\text{w} = 0$, your model is confused and it returns the same value for both of them.
