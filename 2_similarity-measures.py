# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>

#%% [markdown]
# ### Distances
#%% [markdown]
# ![](cosine.png)
#%% [markdown]
# #### Manhattan Similarity

#%%
from math import*
 
def square_rooted(x):
 
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
 
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)
 
print (cosine_similarity([1,1,1], [3,3,3]))

#%% [markdown]
# #### Jaccard Similarity

#%%
from math import*
 
def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
 
print (jaccard_similarity(["banana","orange","grapes"],["apple", "banana", "grapes", "pear"]))

#%% [markdown]
# #### Eucldean Distance

#%%
from math import *


#%%
v1 = [1,1,1]
v2 = [3,3,3]
v3 = [3,0,2,6]


#%%
def eclu_dist(x,y):
    return sqrt(sum(pow(a-b, 2) for a,b in zip(x,y)))


#%%
print(eclu_dist(v1,v2))

#%% [markdown]
# #### Manhatan Distance

#%%
def man_dist(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))


#%%
print(man_dist(v1,v2))
print(man_dist(v1,v3))


#%%
c1 = [1,2]
c2 = [4,3]
c3 = [4,1]


#%%
pt = [2,2]


#%%
print(man_dist(pt, c1))
print(man_dist(pt, c2))
print(man_dist(pt, c3))