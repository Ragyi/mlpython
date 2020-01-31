# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# <h2>Introduction to Python</h2>
#%%
dic = {}
dic['one'] = 'This is one'
dic[2] = 'This is two'
print(dic['one'])
print(dic)
dic['one'] = 'One has changed'
print(dic)


#%%
l = [1, 2, 3, 4, 5, 6, 7, 8]
s = "This is a string."

print(l[1:5])
print(s[0:6])

#%% [markdown]
# #### Branching and Decisioning

#%%
trip1 = 15

if trip1 <= 25:
    print('25 and Under')
else:
    print("25 and Over")


#%%
stat1 = True
stat2 = False

if stat1:
    print("1st stat is true")
elif stat2:
    print("2nd stat true")
else:
    print("Both are false")

#%% [markdown]
# #### Iterations (Loops)
#%% [markdown]
# ##### For Loops

#%%
exampleList = [1, 2, 3, 4, 5]

for i in exampleList:
    print(i)


#%%
#String with dynamic object
x=list(range(2,6))

print("Initial List: {}".format(x))

for idx, i in enumerate(x):
    x[idx]= i**2
    
print("The new list:{}".format(x))

#During each step of the for loop, enumerate(x)iterates through the list 
#and store the index in [idx] and value in [i].


#%%
newList = [x**2 for x in range (2, 6)]
print(newList)

#%% [markdown]
# ##### While Loops

#%%
i = 0
while i < 5:
      print (i, end=" ")   # prints each iteration on the same line
      i += 1            # adds 1 to the variable i
print()                 # prints a blank line
print("done")       # Note that this is printed outside the loop


#%%
y = range(1, 51)

for i in y: 
    if i%3==0 and i%2!=0:
        print(i)

#%% [markdown]
# ##### Functions

#%%
def func1(s):
    "Some stuff"
    
    print("Number of characters in the string: ", len(s))
    return 2*len(s)


#%%
func1("test function")

#%% [markdown]
# ###### Returning multiple values from a function

#%%
def powers(x):
    xs = x**2
    xc = x**3
    xf = x**4
    return xs, xc, xf


#%%
powers(5)


#%%
y1, y2, y3 = powers(5)
print(y2)

#%% [markdown]
# ##### Anonymous Functions
#%% [markdown]
# Anonymous functions are defined by the keyword  lambda  in Python. Functions  f  and  g  in the cell below basically do the same thing. But  g  is an anonymous function.

#%%
# define a function
def f(x):                 
    return x**2.          # x to the power of 2 - the function give us x squared

# use an anonymous function instead
g = lambda x: x**2.  # x to the power of 2 - in other words x squared

print(f(8))   # call the f function and ask for the square of 8
print(g(8))  # call the g anonymous function and ask for the square of 8

#%% [markdown]
# In the cell below, we used anonymous function n_increment(). We create new functions by passing  n  to n_incremenet(). For example  f5  and  f9  are functions that add 5 and 9 to their inputs respectively.

#%%
def n_increment(n):
    return lambda x: x+n
add5 = n_increment(5)
print(add5(2))