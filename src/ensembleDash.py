import numpy as np
import pandas as pd
import json
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import GradientBoostingClassifier as grad
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import VotingClassifier
#import matplotlib.pyplot as plt
#import os
#from os.path import exists
#from os import remove
#from sklearn.model_selection import train_test_split
#import sklearn.model_selection as ms
import sklearn.metrics as sm
#from sklearn.model_selection import cross_validate
#from sklearn import metrics
#import scipy.stats as stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
#import plotly.express as px
#import util as util

# Train the models
with open('../data/processedData/columnDataDictionary.json') as d:
    columnDataDictionary = json.load(d)
with open('../data/processedData/treeParams.json') as d:
    treeParams = json.load(d)
    preciseTreeParams = treeParams["preciseTreeParams"]
    recallTreeParams = treeParams["recallTreeParams"]
with open('../data/processedData/forestParams.json') as d:
    forestParams = json.load(d)
    preciseForestParams = forestParams["preciseForestParams"]
    recallForestParams = forestParams["recallForestParams"]  

datingTrain = pd.read_csv('../data/plotlyDashData/datingTrain.csv')
match = datingTrain["match"]
X = datingTrain.drop("match",axis=1)
datingTest = pd.read_csv('../data/plotlyDashData/datingTest.csv')
matchTest = datingTest["match"]
XTest = datingTest.drop("match",axis=1) 

sqrtn = int(np.sqrt(X.shape[0]))
logModel = lm.LogisticRegression(max_iter=1e9)
logPipe = make_pipeline(StandardScaler(), logModel)
knn5 = knn(n_neighbors=5)
knnsqrtn = knn(n_neighbors=sqrtn)
gradientdeci = grad(learning_rate=0.1)
gradientdeka = grad(learning_rate=10)
preciseTree = tree(criterion = preciseTreeParams["criterion"],
                    max_depth = preciseTreeParams["max_depth"],
                    max_features = preciseTreeParams["max_features"])
recallTree = tree(criterion = recallTreeParams["criterion"],
                  max_depth = recallTreeParams["max_depth"],
                  max_features = recallTreeParams["max_features"])
preciseForest = rf(n_estimators = preciseForestParams["n_estimators"],
                    criterion = preciseForestParams["criterion"],
                    max_depth = preciseForestParams["max_depth"],
                    max_features = preciseForestParams["max_features"])
recallForest = rf(n_estimators = recallForestParams["n_estimators"],
                  criterion = recallForestParams["criterion"],
                  max_depth = recallForestParams["max_depth"],
                  max_features = recallForestParams["max_features"])

estimtatorTuples = [
        ("logModel",logPipe),
        ("knn5",knn5),
        ("knnsqrtn",knnsqrtn),
        ("gradientdeci",gradientdeci),
        ("gradientdeka",gradientdeka),
        ("preciseForest",preciseForest),
        ("recallForest",recallForest)
    ]

ensembleVote = VotingClassifier(estimators = estimtatorTuples)
allEstimators = estimtatorTuples.append(("Ensemble",ensembleVote))

# Dash code
app = Dash(__name__, use_pages=True)

centerMiddle = {
    "align-text":"center",
    "vertical-align":"middle"
}

hidden = {
    "display":"hidden"
}

nostyle = {}

selected = {
    "background-color":"aqua"
}

hover = {
    "color":"blue"
}

sidebar = html.Div(
    [
        html.H2("Profile Parameters"),
        html.Hr(),
        dbc.Nav(
            [
                html.Div(style=centerMiddle,children="Home",id="hometab"),
                html.Div(style=centerMiddle,children="Matches",style = hidden,id="matchestab"),
                html.Div(style=centerMiddle,children="Basics",id="basicstab"),
                html.Div(style=centerMiddle,children="Interests",id="intereststab"),
                html.Div(style=centerMiddle,children="Decision Questions",id="questionstab")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"background-color":"gainsboro"}
)

app.layout = html.Div(children= [
    html.div(style='display:content'),
    html.H1(children='Ensemble Dash!'),
    Dash.page_container,
    html.Footer(children=[
       html.A(children="Learn about our match makers!",style=hover),
       html.Br(),
       dcc.Link(children=html.span(
        children="Or you can play with some of their data"
       ),href="/sandbox"),
       html.Span(children='''
       This product used the Fake Name Generator API but is not endorsed or certified by the Fake Name Generator.
       ''')
    ]),
    dcc.Store(id='confusionMatricies', storage_type='session'),
    dcc.Store(id='matchSlate',storage_type="session"),
    dcc.Store(id='matchResults',storage_type="session")
])

if __name__ == '__main__':
    app.run_server(debug=True)