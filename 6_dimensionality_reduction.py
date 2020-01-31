# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>

#%% [markdown]
# ### PCA

#%%
# import required modules
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


#%%
data = pd.read_csv('/Users/ragyibrahim/Documents/Uni/2019/Trimester 2/SIT720/AT1/data.csv', delimiter=',', header = None).values
print(data.shape)

#%% [markdown]
# #### Plot the data

#%%
plt.plot(data[:,0],data[:,1], '.', markersize=14)
plt.title('Original Data')
plt.show()

#%% [markdown]
# #### Normalise the data

#%%
mu = data.mean(axis=0) # means of each column
sigma = data.std(axis=0) #std of each column
print(mu)
print(sigma)
print("#####")
##Use mean and std to normalise data
Xnorm = ((data-mu)/sigma)
print(Xnorm[0:5,:])

#%% [markdown]
# #### Compute covariance matrix of normalised data
#%% [markdown]
# If m is the number of training data, calculate the covariance matrix as:  
# $\sum_{}^{} =\frac{1}{m}Xnorm^TXnorm$

#%%
m = len(Xnorm)
covmatrix = np.dot(Xnorm.T, Xnorm)/m
print(covmatrix)

#%% [markdown]
# #### Compute Eigenvectors and Eigenvalues
#%% [markdown]
# Now, compute the eigenvalues(S) and eigenvectors (U) of this covariance matrix. The eigenvectors(U) become the principal components. We use **linalg.eig()** function from *numpy* to compute the eigenvalues and eigenvectors of a square array.

#%%
S,U = np.linalg.eig(covmatrix) #The linalg.eig() function returns 2 arrays. The first is the eigenvalues and the second contains the corresponding eignevectors. Therefore by doing S,U = np.linalg.eig(covmatrix) we will store the first array (eigenvalues) in S and the second array (eigenvectors) in U
print('Eignevalues: {}'.format(S))
print('Eignevectors: ')
print(U)

#%% [markdown]
# #### Project data points unto new principle axis (Eigenvectors)
#%% [markdown]
# This is done by a simple linear transformation

#%%
Z = np.dot(Xnorm, U)
print(Z[0:5,:])
print('#############')
print(Xnorm[0:5,:])

#%% [markdown]
# #### Lets visualise the data before and after PCA

#%%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,5)) #Create side-by-side plots
fig.subplots_adjust(wspace=0.2) # leave some space between figs

#Plot the original data
axs[0].scatter(data[:,0], data[:,1])
axs[0].set_title("Original data")

#Plot Uncorrelated data
axs[1].scatter(Z[:,0], Z[:,1])
axs[1].set_title("Uncorrelated Data")

#%% [markdown]
# #### Dimensionality Reduction - Of uncorrelated data
#%% [markdown]
# To reduce the dimensionality of our $2D$ data to $1D$, we remove the principle component that captures the least variation. Our principle components, which are the eigen vectors of the covariance matrix are: $U[:,0]$ and $U[:,1]$. By projecting our data Xnorm onto just $U[:,0]$, we get a reduced $Z$ in $1D$.
# 
# In general, we decide to keep k eigenvectors in $U$ that captures maximum variation. Then our reduced data $Znorm$ becomes:
# 
# $\text{Xnorm}_{M\times M} \times \text{Ureduced}_{M\times k}=\text{Z}_{M\times k}$

#%%
#In the case where we want k=1 (i.e. reduce rank to 1)

k=1 # number of principal components to retain
U_red = U[:,0:k] # choose the first k principal components
#project our data Xnorm onto reduced U
Z_red = np.dot(Xnorm, U_red)
print(Z_red.shape)
print(U_red.shape)

#%% [markdown]
# #### Visualise data after dimensionality reduction

#%%
#Project our Xnorm data Using Z_red
X_rec = np.dot(Z_red, U_red.T)
#Visualize the recovered data
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,5))  
fig.subplots_adjust(wspace=0.2) # leave some space between figs

#Plot Xnorm data
axs[0].scatter(Xnorm[:,0], Xnorm[:,1])
axs[0].set_title("Normalised Original Data")

#Plot For "Recovered" Data
axs[1].scatter(X_rec[:,0], X_rec[:,1])
axs[1].set_title("Data after Dimensionality reduction")

#%% [markdown]
# #### Measuring ‘reconstruction error’
#%% [markdown]
# How much information did we lose after dimensionality reduction? To measure this, we calculate the reconstruction error of our data. Reconstruction error, is calculated as the square root of sum of squared errors of each data point. Essentially, this becomes the distance between the original data point and the reconstructed data point.
# 
# $ \text{Reconstruction Error}= \sqrt\ \left( \sum_{i=1}^{M}\sum_{j=1}^{N}(\text{Xnorm}_{ij}-\text{Xrec}_{ij})^{2}\right)$
# 
# ###### Forbenius Norm
# To make this error term between  0  and  1 , we divide it by the Frobenius norm of the original data Xnorm. Frobenius norm of a matrix is defined as the square root of the sum of the absolute squares of its elements.
# 
# In python, frobenius norm is implemented in linear algebra package of numpy. You can call it using linalg.norm(, 'fro')

#%%
recon_error = np.linalg.norm(Xnorm-X_rec, 'fro')/np.linalg.norm(Xnorm, 'fro')
print("The reconstruction error is: {}".format(recon_error))

#%% [markdown]
# ### PCA Using Inbuilt Modules
#%% [markdown]
# ##### Import data

#%%
newData = pd.read_csv('/Users/ragyibrahim/Downloads/5DGaussian.csv', delimiter= ',', header=None).values
print(newData[0:5,:])
print(newData.shape)

#%% [markdown]
# #### Normalise data

#%%
from sklearn.preprocessing import scale
Xnorm_py = scale(newData)

#%% [markdown]
# #### Implement PCA

#%%
from sklearn.decomposition import PCA
newDataPca = PCA(n_components=5)
newDataPca.fit(Xnorm_py)

#%% [markdown]
# #### Dimensionality Reduction
#%% [markdown]
# ###### Dimensionality Reduction: How to choose the dimensions to keep
# To put it differently, how can we choose the number of principal components to retain? We can decide this by looking at the variance captured by each principle component.

#%%
#Find the variance captured by each Principle Axis
newDataVar = newDataPca.explained_variance_ratio_
print(newDataVar)

#%% [markdown]
# From the above array, we see that the first component captures around  84%  variance, second component captures around  10%  variance and so on. To make it much easier, we can calculate the cumulative variance:

#%%
newDataCumVar = np.cumsum(np.round(newDataVar, decimals=4)*100)
print(newDataCumVar)
plt.plot(newDataCumVar)
plt.xlabel("Number of Principle Axis")
plt.ylabel("Cumilative Variance Captured")

#%% [markdown]
# So, if k is the number of principal components, we see that  k=1  captures around  84%  variance,  k=2  (the first  2  components together) capture around  94%  variance and so on. Since  k=2  captures more than  90% variance in our data, lets drop the third, fourth and fifth components.

#%%
newDataPca2 = PCA(n_components=2)
newDataTrans = pca.fit_transform(Xnorm_py) #this fits the models and applies the reduction
print(Zred.shape)

#%% [markdown]
# #### Compute reconstruction error
#%% [markdown]
# We can recreate our original data (Xrec) from the reduced data (Zred) using the inverse_transform() function, and calculate the reconstruction error as before.

#%%
newDataRcov = pca.inverse_transform(newDataTrans)
print(newDataRcov.shape)

#%% [markdown]
# #### Influence of Dimensionality Reduction on Reconstruction error
#%% [markdown]
# Let us see how dropping the dimensionality of data affects the reconstruction error. We perform PCA using increasing number of principal components, and measure the reconstruction error in each case.

#%%
nSamples, nDims = Xnorm_py.shape

# vary principal components from 1 to 5
n_comp = range(1,nDims+1)
print(n_comp)


#%%
# Initialize vector of rec_error
rec_error = np.zeros(len(n_comp)+1)

for k in n_comp:
    pca = PCA(n_components=k)
    Zred = pca.fit_transform(Xnorm_py)
    Xrec = pca.inverse_transform(Zred)
    rec_error[k] = np.linalg.norm(Xnorm_py-Xrec, 'fro')/np.linalg.norm(Xnorm_py, 'fro')
    print("k={}, rec_error={}".format(k, rec_error[k]))

rec_error = rec_error[1:] #we started recording from index 1, so drop index 0


#Visualize the change in error
plt.plot(n_comp,rec_error)
plt.xlabel('No of principal components (k)')
plt.ylabel('Reconstruction Error')