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
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
import dash
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

datingTrain = pd.read_csv('../data/plotlyDashData/datingTrain.csv')
datingTest = pd.read_csv('../data/plotlyDashData/datingTest.csv')

match = datingTrain["match"]
X = datingTrain.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])
matchTest = datingTest["match"]
XTest = datingTest.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])


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
allEstimatorTuples = [("Ensemble",ensembleVote)] + originalEstimtatorTuples
for estimatorTuple in allEstimatorTuples:
    (estimatorTuple[1]).fit(X,match)

# Dash code
app = Dash(__name__, use_pages=True)

middleAndCenter = styles.middleAndCenter
hidden = styles.hidden
nostyle = styles.nostyle
selected = styles.selected
unselected = styles.unselected
fitContent = styles.fitContent
sidebarstyle = styles.SIDEBAR_STYLE

sidebar = html.Div(style=sidebarstyle,children=[
    dbc.Nav(
        [
            dbc.NavLink(
                [
                    html.Div(page["name"], className="ms-2"),
                ],
                href=page["path"],
                active="exact"
            ) for page in dash.page_registry.values()
        ],
        vertical=True,
        pills=True
    )
])

app.layout = html.Div(children= [
    sidebar,
    html.Div(children=[
        html.H1(children='Ensemble Dash!',style={"background-color":"dodgerblue"}),
        html.H2(id="pagetitle",children='Sandbox',style={"background-color":"dodgerblue"}),
        dcc.Checklist(
            id="modelSelection",
            options=[estimatorTuple[0] for estimatorTuple in originalEstimtatorTuples],
            value=[estimatorTuple[0] for estimatorTuple in originalEstimtatorTuples]
        ),
        dash.page_container
    ])
    
])
print("I'm hit")

#modelSection callback
@dash.callback(
    Output('EnsembleMatrix', 'figure'),
    Output('EnsembleMetrics', 'figure'),
    Output('logModelInfo', 'style'),
    Output('knn5Info', 'style'),
    Output('knnsqrtnInfo', 'style'),
    Output('gradientdeciInfo', 'style'),
    Output('gradientdekaInfo', 'style'),
    Output('recallTreeInfo', 'style'),
    Output('preciseTreeInfo', 'style'),
    Output('recallForestInfo', 'style'),
    Output('preciseForestInfo', 'style'),
    Input('modelSelection', 'value'))
def updateEnsemble(models):
    includedModels = []
    styleValues = []

    if len(models)==0:
        models = [modelTuple[0] for modelTuple in originalEstimtatorTuples]

    if ("logModel" in models):
        premod = lm.LogisticRegression(max_iter=1e9)
        mod = make_pipeline(StandardScaler(), logModel)
        includedModels.append(("logModel",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("knn5" in models):
        mod = knn(n_neighbors=5)
        includedModels.append(("knn5",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("knnsqrtn" in models):
        mod = knn(n_neighbors=sqrtn)
        includedModels.append(("knnsqrtn",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("gradientdeci" in models):
        mod = grad(learning_rate=0.1)
        includedModels.append(("gradientdeci",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("gradientdeka" in models):
        mod = grad(learning_rate=10)
        includedModels.append(("gradientdeka",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("recallTree" in models):
        mod = make_pipeline(StandardScaler(), logModel)
        includedModels.append(("recallTree",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("preciseTree" in models):
        mod = make_pipeline(StandardScaler(), logModel)
        includedModels.append(("preciseTree",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("recallForest" in models):
        mod = make_pipeline(StandardScaler(), logModel)
        includedModels.append(("recallForest",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    if ("preciseForest" in models):
        mod = make_pipeline(StandardScaler(), logModel)
        includedModels.append(("preciseForest",mod))
        styleValues.append(nostyle)
    else:
        styleValues.append(hidden)
    
    newEnsemble = VotingClassifier(estimators = includedModels)
    newEnsemble.fit(X,match)
    ypredict = newEnsemble.predict(XTest)
    confusionMatrix = confusion_matrix(matchTest,ypredict)
    accuracyScore = accuracy_score(matchTest,ypredict)
    recallScore = recall_score(matchTest,ypredict)
    precisionScore = precision_score(matchTest,ypredict)

    cm = px.imshow(confusionMatrix,
    labels=dict(x="Actual", y="Predicted", color="Productivity"),
                x=['Match Fail', 'Match Success'],
                y=['Match Fail', 'Match Success'],
    text_auto=True)
    metrics=px.bar(x=[accuracyScore,recallScore,precisionScore],y=["accuracy","recall","precision"],
                orientation='h',hover_data=["accuracy","recall","precision"])

    return cm,metrics,styleValues[0],styleValues[1],styleValues[2],styleValues[3],styleValues[4],styleValues[5],styleValues[6],styleValues[7],styleValues[8]

#sandbox callbacks

#run app
if __name__ == '__main__':
    app.run_server(debug=True)