# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# <h2>Logistic regression in python</h2>

#%%
#Load data
examData = pd.read_csv('/Users/ragyibrahim/Downloads/ExamScores.csv')
rows, cols = examData.shape
print('Data has {} rows and {} cols'.format(rows, cols))
examData.head()


#%%
#Split data into trainig (70%) and testing (30%)
from sklearn.model_selection import train_test_split

trainData, testData = train_test_split(examData, test_size = 0.3)
print(trainData.shape)
print(testData.shape)


#%%
#Split dependant and independant variables
examPredictors = ['Exam1', 'Exam2']
examResponse = ['Admit']

print(trainData[predictors].head())
print()
print(testData[predictors].head())

#%% [markdown]
# <h4>Regularisation using $L_1$ or $L_2$</h4>
# 
# You can perform regularised logistic regression by specifying two arguments in the function call:
# 
# - penalty: this takes values **l1** for lasso and **l2** for ridge regression
# - C: this is the inverse of regularisation parameter alpha or lambda. smaller values specify stronger regularisation
# 
# *You can refer to the documentation for more detailed information*
# 
# Let’s try with $\lambda = 0.1$. So we have to set $C = \frac{1}{0.1}$

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Define hyperparameters
lamdaValue = 0.1
C = 1/lamdaValue
#Fit model
logRegModel = LogisticRegression(C = C, penalty = 'l2')
logRegModel.fit(trainData[predictors], trainData[response])
yPredictLogRidge = logRegModel.predict(testData[predictors])


#%%
#Evaluate the model performance
logRegAcc = accuracy_score(yPredictLogRidge, testData[response])
print('Model Accuracy is: {}'.format(logRegAcc))
print("Model Coeff: {}".format(np.append(logRegModel.intercept_, logRegModel.coef_)))

#%% [markdown]
# Now as an exercise let’s do the following. For a list of l1 and l2 penalty scores, lets calculate the average accuracy over 500 runs of L1 and L2 regularised Logistic regression.
# 
# Let’s begin by defining a function that takes in data and penalty types and values. For a fixed number of runs (‘trials’), the data is randomly split 70/30 as train/test. The average test accuracy is calculated.

#%%
def runLogRegModel(trials, data, predictors, label, penType, penScore):
    #define variables
    modelAcc = 0
    modelWeight = np.zeros([1,3])
    #For loop
    for i in range(0,trials):
        #Fit Model to Training data
        trainData, testData = train_test_split(data, test_size = 0.3)
        logRegModel = LogisticRegression(C = 1/penScore, penalty = penType)
        logRegModel.fit(trainData[predictors], trainData[label])
        yPredict = logRegModel.predict(testData[predictors])
        #Evaluate Model Accuracy
        modelAcc += accuracy_score(yPredict, testData[label]) #appends scores
        modelWeight += np.append(logRegModel.intercept_, logRegModel.coef_)
    #Average scores
    modelAcc /= trials
    modelWeight /= trials
    #Function will Return model accuracy and weights     
    return np.round(modelAcc, decimals = 2), np.round(modelWeight, decimals = 2)

#%% [markdown]
# <h5>$L_2$ Regulisation</h5>
# 
# Using the above function lets now try to find the best **lambda** value from 500 random splits of our data:

#%%
#Define set of lambda values to iterate over
lambdaValues = [.0001,.0003,.001,.003,.01,.03,.1,.3,1,3,5,10]
l2Acc = np.zeros(len(lambdaValues))
index = 0

#L2 reg
for i in lambdaValues:
    l2Acc[index], w = runLogRegModel(500, examData, examPredictors, 'Admit', 'l2', np.float(i))
    index += 1
#Print accuracy for each lambda    
print("Acc: {}".format(l2Acc))
#Penalty at which validation accuracy is max
maxAccIndexL2 = np.argmax(l2Acc)
#Find corresponsing optimal lambda value
optiLambda = lambdaValues[maxAccIndexL2]
print("Optimal Lambda value: {}".format(optiLambda))

#%% [markdown]
# <h5>$L_1$ Regulisation</h5>
# 
# Using the above function lets now try to find the best **alpha** value from 500 random splits of our data:

#%%
#Define set of lambda values to iterate over
alphaValues = [.0001,.0003,.001,.003,.01,.03,.1,.3,1,3,5,10]
l1Acc = np.zeros(len(alphaValues))
index = 0

#L1 reg
for i in alphaValues:
    l1Acc[index], w = runLogRegModel(500, examData, examPredictors, 'Admit', 'l1', np.float(i))
    index += 1
#Print accuracy for each alpha    
print("Acc: {}".format(l1Acc))
#Penalty at which validation accuracy is max
maxAccIndexL1 = np.argmax(l1Acc)
#Find corresponsing optimal alpha value
optiAlpha = alphaValues[maxAccIndexL1]
print("Optimal Alpha value: {}".format(optiAlpha))

#%% [markdown]
# <h5>Comparing $L_2$ and $L_1$ metrics</h5>
# 
# We now plot the average model accuracy with respect to the parameters alpha and lambda. Also, 
# we mark the best values of alpha and lambda.

#%%
#plot the accuracy curve
plt.plot(range(0,len(lambdaValues)), l2Acc, color='b', label='L2')
plt.plot(range(0,len(alphaValues)), l1Acc, color='r', label='L1')
#replace the x-axis labels with penalty values
plt.xticks(range(0,len(lambdaValues)), lambdaValues, rotation='vertical')

#Highlight the best values of alpha and lambda
plt.plot((maxAccIndexL2, maxAccIndexL2), (0, l2Acc[maxAccIndexL2]), ls='dotted', color='b')
plt.plot((maxAccIndexL1, maxAccIndexL1), (0, l1Acc[maxAccIndexL1]), ls='dotted', color='r')

#Set the y-axis from 0 to 1.0
axes = plt.gca()
axes.set_ylim([0, 1.0])

plt.legend(loc="lower left")
plt.show()


#%%
weights = np.append(logRegModel.intercept_, logRegModel.coef_)


#%%
weights.shape
#%%
#iterate over items in a list
#check if df has variables of type np.float64 - bool returns TRUE or FALSE
[(e,type(e), isinstance(e, (int, np.float64))) for e in df]