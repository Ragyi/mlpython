# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>

#%% [markdown]
# #### Scikit-Learn Package for ML

#%%
import sys
get_ipython().system(u'{sys.executable} -m pip install -U scikit-learn')


#%%
from sklearn import datasets


#%%
digits = datasets.load_digits()
print(digits)


#%%
print(digits.target)

#%% [markdown]
# Linear regression using SciKit-learn

#%%
from sklearn.linear_model import LinearRegression
import numpy as np

#Generate training data
X = np.random.rand(100, 1)
Y = np.exp(X)

#Create linear model
linearModel = LinearRegression()
#Fit linear model to training data
linearModel.fit(X,Y)

#Generate test data
X_test = np.random.rand(1000,1)
Y_test = linearModel.predict(X_test)

plt.plot(X_test,Y_test, ".r")
plt.plot(X,Y, ".b")
plt.show()

#%% [markdown]
# #### Text analysis with TF-IDF score
#%% [markdown]
# Creating the document

#%%
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

print ("The corpus is the list: {}".format(corpus))

#%% [markdown]
# Using CountVectorizer in SciKit we can implement tozenisation and occurrence counting in a single class:
# 
# Load module for sklearn:

#%%
from sklearn.feature_extraction.text import CountVectorizer

#%% [markdown]
# Initiate module:

#%%
vectoriser = CountVectorizer()
vectoriser

#%% [markdown]
# Create the term freq matrix:

#%%
termFreq = vectoriser.fit_transform(corpus)
vectoriser.get_feature_names()

#%% [markdown]
# Now we can transform the term freq output into an array:

#%%
termFreq.toarray()

#%% [markdown]
# To do a TF-IDF transformation, we use TfidfTransformer:

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
TFvector = TfidfVectorizer()
TFvector


#%%
#Apply to corpus:
tfVectorisation = TFvector.fit_transform(corpus)
tfVectorisation.toarray()

#%% [markdown]
# By default, the tf-idf vectorization returns a sparse matrix. We can see the output by converting it to a dense matrix with:

#%%
print(vectoriser.vocabulary_)
tfVectorisation.todense()


#%%
import sys
get_ipython().system(u'{sys.executable} -m pip install wordcloud')


#%%
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

SOME_TEXT = "A tag cloud is a visual representation for text data, typicallyused to depict keyword metadata on websites, or to visualize free form text. Some more text and tag."

wordcloud = WordCloud(stopwords = STOPWORDS, background_color= 'white', width = 1200,
                     height = 1000).generate_from_text(SOME_TEXT)

print(wordcloud.words_)
fig = plt.figure()
plt.imshow(wordcloud)
plt.show()