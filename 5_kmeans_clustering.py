# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# #### Kmeans clustering in Python
#%% [markdown]
# Load required mdoules

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%% [markdown]
# Create sample dataset

#%%
x = [1,5,1.5,8,1,9]
y=[2,8,1.8,8,0.6,11]

#%% [markdown]
# Plot data

#%%
plt.scatter(x,y)
plt.show()

#%% [markdown]
# Now lets create a matrix X with x,y coordinates

#%%
X = np.array([
    [1,2],
    [5,8],
    [1.5, 1.8],
    [8,8],
    [1,0.6],
    [9,11]
])

#%% [markdown]
# Initiate K-Means algorithm with 2 clusters

#%%
kmeans_1 = KMeans(n_clusters=2)
kmeans_1.fit(X)

#%% [markdown]
# Now, we have fit the KMeans model to our data, X. The model will have identified 2 clusters, with 2 cluster centres (centroids). we can get this data as:

#%%
centroids = kmeans_1.cluster_centers_
labels = kmeans_1.labels_

print(centroids)
print(labels)

#%% [markdown]
# Lets try to visualise the clusters by plotting them. The centroids will be marked as “X”

#%%
colors = ['g.', 'r.', 'c.', 'y.']

for i in range(len(X)):
    print("coordinates: ", X[i], "label: ", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
#Visualise the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker = "X", s = 150, linewidths = 1, zorder = 10 )
plt.show()


#%%
import numpy as np


#%%
x=np.array([[1,0],[0,-1]])
y=np.array([[-3,-2], [-4,1], [0,4], [4,1], [2,-3]])


z= np.dot(y,x)
print(z)
