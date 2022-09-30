#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import os
#from os.path import exists
#from os import remove
#import json
#from sklearn.model_selection import train_test_split
#import sklearn.model_selection as ms
#import sklearn.metrics as sm
#from sklearn.model_selection import cross_validate
#import sklearn.linear_model as lm
#from sklearn.neighbors import KNeighborsClassifier as knn
#from sklearn.ensemble import GradientBoostingClassifier as grad
#from sklearn.ensemble import RandomForestClassifier as rf
#from sklearn.ensemble import VotingClassifier
#from sklearn import metrics
#import scipy.stats as stats
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#import util as util
import webbrowser
from dash import Dash, html, dcc
import plotly.express as px

# Dash code
app = Dash(__name__, use_pages=True)

app.layout = html.Div(children= [
    html.div(style='display:content'),
    html.H1(children='Ensemble Dash!'),
    Dash.page_container,
    html.Footer(children=[
       dcc.Link(children=html.span(
        children="Learn about our match makers!"
       ),href="/matchmakers") 
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
    webbrowser.open('http://127.0.0.1:8050/edit')