# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#<h1> SIT 720 - Python Intro </h1>
#%% [markdown]
# #### Using Plotly for charts

#%%
import sys
get_ipython().system(u'{sys.executable} -m pip install scipy')


#%%
import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


#%%
import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets
from scipy import special


#%%
#create the data
randomData = np.linspace(0, np.pi, 1000)


#%%
#Create plotly layout
layout = go.layout(
    title = "Simple Graph",
    yaxis = dict(
        title = 'volts'
    ),
    xaxis = dict(
        title = 'nanoseconds'
    )
)
##Create your graph setup as a trace
trace1 = go.Scatter(
    x = randomData,
    y = np.sin(randomData), 
    mode = 'lines',
    name = 'Sine(x)',
    line = dict(
        shape = 'spline'
    )
)
##Create your figure
fig = go.Figure(data = [trace1], layout=layout)
fig.show()


#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Make figure with subplots
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"},
                                            {"type": "surface"}]])

# Add bar traces to subplot (1, 1)
fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=1)
fig.add_trace(go.Bar(y=[3, 2, 1]), row=1, col=1)
fig.add_trace(go.Bar(y=[2.5, 2.5, 3.5]), row=1, col=1)

# Add surface trace to subplot (1, 2)
# Read data from a csv
z_data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv")
fig.add_surface(z=z_data)

# Hide legend
fig.update_layout(
    showlegend=False,
    title_text="Default Theme",
    height=500,
    width=800,
)

fig.show()
