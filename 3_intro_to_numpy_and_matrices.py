# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# #### Intro to Numpy Module

#%%
import numpy as np
#%%
a = np.random.randn(5,1)
print(a.shape)
print(a)


#%%
a_trans = a.T
print(a_trans.shape)
print(a_trans)


#%%
a_dot = np.dot(a, a_trans)
print(a_dot.shape)
print(a_dot)

#%% [markdown]
# #### Using MatPlot in Python

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% [markdown]
# Create to arrays X and Y

#%%
x = np.array([1,2,3,4])
y = np.array([10,12,33,56])

xx = np.array([1.5,2.5,3.5])
yy = np.array([10.5, 15.5, 20])

#%% [markdown]
# plot x and y

#%%
plt.plot(x,y, '*r') # The plt.plot function takes the argument for plot type *r = red stars
plt.show() # By having a plt.show() for each plot this will create 2 seperate plots
plt.plot(xx,yy, '.b') # This will overlay the current plot with blue dots (.b)
plt.show() # This line shows the plot

#%% [markdown]
# #### Extending Numpy

#%%
A = np.array([(1,2), (3,4)])
print(A)

#%% [markdown]
# An all zero matrix

#%%
B = np.zeros([3,3])
print(B)

#%% [markdown]
# All 1 matrix

#%%
C = np.ones([3,3])
print(C)

#%% [markdown]
# Identity Matrix

#%%
D = np.identity(3)
print(D)

#%% [markdown]
# Matrix of random numbers

#%%
E = np.random.randn(4,3)
print(E)

#%% [markdown]
# ##### Adding or subtracting a scalar value to a matrix 

#%%
print(A)
print()
print("After addition of a scalar: ")
print(A+3)

#%% [markdown]
# ##### Adding or subtracting matrices

#%%
aa = np.identity(2)
bb = np.random.randn(2,2)
print("Matrix AA")
print(aa)
print("Matrix BB")
print(bb)

#%% [markdown]
# Lets add aa and bb together

#%%
result = aa + bb
print(result)

#%% [markdown]
# ##### Multiplying matrices

#%%
ac = np.random.randn(3,3)
ca = np.random.randn(3,2)


#%%
print(np.shape(ac))
print(np.shape(ca))


#%%
print(ac.dot(ca))
print("+++++++++++++++++++++++++++++++++++++++")
print(np.dot(ac,ca))

#%% [markdown]
# The otherway around does not work as the coulmns of the first is not equal to the rows of the second

#%%
print(np.dot(ca,ac))

#%% [markdown]
# ![](inversematrix.png)

#%%
cc = np.random.randn(3,3)
cc_inverse = np.linalg.inv(cc)
print("This is the original matrix:")
print(cc)
print("This is it's inverse:")
print(cc_inverse)

#%% [markdown]
# Now let's check if the condition holds up:

#%%
print(np.dot(cc, cc_inverse)) #should produce an identity matrix
print("Which is also identical to: ")
print(np.dot(cc_inverse, cc))

#%% [markdown]
# #### Transposing a Matrix

#%%
AA = np.arange(6).reshape(3,2)
BB = np.arange(8).reshape(2,4)
print(AA)
print("===========")
print(BB)

#%% [markdown]
# Transpose of A

#%%
print(AA.T)

#%% [markdown]
# A note: Let matrix  A  be of dimension  n×m  and let  B  be of dimension  m×p. Then  (AB)′=B′A′

#%%
print(np.dot(AA,BB).T)


#%%
print(np.dot(BB.T, AA.T))


#%%
print(A)
print("This is the first column of the matrix A: ")
print(A[:,0])


#%%
print(A[-1,1])

#%% [markdown]
# Using logical checks to extract values from matrices:

#%%
#give the element in the last column that is greater than 3
print(A[:,1]>3)

#%% [markdown]
# Create a  12×2  matrix and print it out:

#%%
A = np.arange(24).reshape(12,2)
print(A)


#%%
for i in A:
    print(i)


#%%
for j in A.T:
    print(j)

#%% [markdown]
# #### Find the minimum of a function

#%%
import numpy as np
from scipy.optimize import fmin
import math


#%%
## Define the function

def f(x):
    val = math.pow(x,2) +1
    return val

funMin = fmin(f,np.random.randn(1,1))
print(funMin)