import numpy as np
import pandas as pd
import json
import sklearn.linear_model as lm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import GradientBoostingClassifier as grad
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import VotingClassifier
from geopy.distance import great_circle

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import styles

#import matplotlib.pyplot as plt
#import os
#from os.path import exists
#from os import remove
#from sklearn.model_selection import train_test_split
#import sklearn.model_selection as ms
#import sklearn.metrics as sm
#from sklearn.model_selection import cross_validate
#from sklearn import metrics
#import scipy.stats as stats
#from dash import Dash, html, dcc, Input, Output
#import util as util

# Train the models
with open('../data/processedData/columnDataDictionary.json') as d:
    columnDataDictionary = json.load(d)
    columnList = columnDataDictionary['columnList']
    nonBinaryCategoricalList = columnDataDictionary['nonBinaryCategoricalList']
    stringToFloatList = columnDataDictionary['stringToFloatList']
    pointDistributionList = columnDataDictionary['pointDistributionList']
    sharedList = columnDataDictionary['sharedList']
    partnerList = columnDataDictionary['partnerList']
with open("../data/processedData/dummyDictionary.json") as d:
    dummyDictionary = json.load(d)
with open('../data/processedData/treeParams.json') as d:
    treeParams = json.load(d)
    preciseTreeParams = treeParams["preciseTreeParams"]
    recallTreeParams = treeParams["recallTreeParams"]
with open('../data/processedData/forestParams.json') as d:
    forestParams = json.load(d)
    preciseForestParams = forestParams["preciseForestParams"]
    recallForestParams = forestParams["recallForestParams"]
with open("../data/descriptionDictionary.json") as d:
    descriptionDictionary = json.load(d)  

datingTrain = pd.read_csv('../data/processedData/datingTrain.csv')
datingTest = pd.read_csv('../data/processedData/datingTest.csv')

dropList = ["iid","pid","round","order","undergra","from","zipcode","dec"]
featureSelectValues = [col for col in columnList if col not in dropList]
featureSelectOptions = [
    {"label":descriptionDictionary[col],"value":col} for col in featureSelectValues
]
candidateDummyList = [str(k) for k in dummyDictionary.keys()]
partnerDummyList = [str(k)+"_o" for k in dummyDictionary.keys()]

candidateFeatures = []
partnerFeatures = []
for col in columnList:
    if col in partnerList:
        candidateFeatures.append(str(col))
        partnerFeatures.append(str(col)+"_o")

#Delete after first run v
questionDictionary = dict()
questionDictionary["1_1"]="What do you what do you look for in a partner? (budget out of 100 points)"
questionDictionary["4_1"]="What do you what do you think others of your gender look for in a partner? (budget out of 100 points)"
questionDictionary["2_1"]="What do you what do you think others of the opposite gender look for in a partner? (budget out of 100 points)"
with open("../data/questionDictionary.json","w") as fp:
    json.dump(questionDictionary,fp)

dummyValueDictionary = dict()
for k in dummyDictionary.keys():
    dummyCategoryDictionary = dict()
    for dummyCol in dummyDictionary[k]:
        print(dummyCol)
        dummyValue = Input()
        dummyCategoryDictionary[dummyCol]=dummyValue
    dummyValueDictionary[k] = dummyCategoryDictionary
with open("../data/dummyValueDictionary.json","w") as fp:
    json.dump(dummyValueDictionary,fp)
#Delete after first run ^

match = datingTrain["match"]
X = datingTrain.drop("match",axis=1)
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

originalEstimtatorTuples = [
        ("logModel",logPipe),
        ("knn5",knn5),
        ("knnsqrtn",knnsqrtn),
        ("gradientdeci",gradientdeci),
        ("gradientdeka",gradientdeka),
        ("preciseTree",preciseTree),
        ("recallTree",recallTree),
        ("preciseForest",preciseForest),
        ("recallForest",recallForest)
    ]

ensembleVote = VotingClassifier(estimators = originalEstimtatorTuples)

ensembleVote.fit(X,match)

# Dash code
app = Dash(__name__, use_pages=True)

centerMiddle = styles.centerMiddle
hidden = styles.hidden
nostyle = styles.nostyle
selected = styles.selected
unselected = styles.unselected
hover = styles.hover
oneNinth = styles.oneNinth
fitContent = styles.fitContent

sidebar = html.Div(
    [
        html.H2("Profile Parameters"),
        html.Hr(),
        dbc.Nav(
            [
                html.Div(style=centerMiddle,children="Sandbox",id="sandboxtab",style=selected),
                html.Div(style=centerMiddle,children="Matchmakers",id="matchesmakerstab")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"background-color":"gainsboro"}
)

app.layout = html.Div(children= [
    html.Div(style='display:content'),
    html.H1(children='Ensemble Dash!',style={"background-color":"dodgerblue"}),
    html.H2(id="pagetitle",children='Sandbox',style={"background-color":"dodgerblue"}),
    dcc.CheckList(
        options=[estimatorTuple[0] for estimatorTuple in originalEstimtatorTuples],
        value=[estimatorTuple[0] for estimatorTuple in originalEstimtatorTuples]
    ),
    Dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)