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
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
import dash
from dash import Dash, html, dcc, Input, Output
import styles
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

datingTrain = pd.read_csv('../data/plotlyDashData/datingTrain.csv')
datingTest = pd.read_csv('../data/plotlyDashData/datingTest.csv')

match = datingTrain["match"]
X = datingTrain.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])
matchTest = datingTest["match"]
XTest = datingTest.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])

with open('../data/processedData/treeParams.json') as d:
    treeParams = json.load(d)
    preciseTreeParams = treeParams["preciseTreeParams"]
    recallTreeParams = treeParams["recallTreeParams"]
with open('../data/processedData/forestParams.json') as d:
    forestParams = json.load(d)
    preciseForestParams = forestParams["preciseForestParams"]
    recallForestParams = forestParams["recallForestParams"]
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
selected = styles.selected
unselected = styles.unselected
fitContent = styles.fitContent
sidebarstyle = styles.SIDEBAR_STYLE
contentstyle = styles.CONTENT_STYLE
displayHidden = styles.displayHidden
displayBlock = styles.displayBlock

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
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(style=contentstyle,children=[
        html.H1(children='Ensemble Dash!',style={"background-color":"dodgerblue"}),
        html.H2(id="pagetitle",style={"background-color":"dodgerblue"}),
        dcc.Checklist(
            id="modelSelection",
            options=[estimatorTuple[0] for estimatorTuple in originalEstimtatorTuples],
            value=[estimatorTuple[0] for estimatorTuple in originalEstimtatorTuples]
        ),
        dash.page_container
    ])
    
])

@dash.callback(
    Output('pagetitle','children'),
    Input('url', 'pathname'))
def updatePageTitle(pathname):
    if pathname.lower() == "sandbox" or pathname.lower() == "/":
        return "Sandbox"
    return "Matchmakers"

#modelSection callback
@dash.callback(
    Output('EnsembleMatrix', 'figure'),
    Output('EnsembleMetrics', 'figure'),
    Output('EnsembleInfo','style'),
    Output('logModelInfo','style'),
    Output('knn5Info','style'),
    Output('knnsqrtInfo','style'),
    Output('gradientdeciInfo','style'),
    Output('gradientdekaInfo','style'),
    Output('recallTreeInfo','style'),
    Output('preciseTreeInfo','style'),
    Output('recallForestInfo','style'),
    Output('preciseForestInfo','style'),
    Input('modelSelection', 'value'))
def updateEnsembleInfo(models):
    includedModels = []
    sVs = []
    if len(models)==0:
        models = [modelTuple[0] for modelTuple in originalEstimtatorTuples]
        sVs.append(displayHidden)
    elif len(models)==1:
        sVs.append(displayHidden)
    else:
        sVs.append(displayBlock)
    if ("logModel" in models):
        premod = lm.LogisticRegression(max_iter=1e9)
        mod = make_pipeline(StandardScaler(), premod)
        includedModels.append(("logModel",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("knn5" in models):
        mod = knn(n_neighbors=5)
        includedModels.append(("knn5",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("knnsqrtn" in models):
        mod = knn(n_neighbors=sqrtn)
        includedModels.append(("knnsqrtn",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("gradientdeci" in models):
        mod = grad(learning_rate=0.1)
        includedModels.append(("gradientdeci",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("gradientdeka" in models):
        mod = grad(learning_rate=10)
        includedModels.append(("gradientdeka",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("recallTree" in models):
        mod = tree(criterion = recallTreeParams["criterion"],
        max_depth = recallTreeParams["max_depth"],
        max_features = recallTreeParams["max_features"])
        includedModels.append(("recallTree",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("preciseTree" in models):
        mod = tree(criterion = preciseTreeParams["criterion"],
        max_depth = preciseTreeParams["max_depth"],
        max_features = preciseTreeParams["max_features"])
        includedModels.append(("preciseTree",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("recallForest" in models):
        mod = rf(n_estimators = recallForestParams["n_estimators"],
        criterion = recallForestParams["criterion"],
        max_depth = recallForestParams["max_depth"],
        max_features = recallForestParams["max_features"])
        includedModels.append(("recallForest",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    if ("preciseForest" in models):
        mod = rf(n_estimators = preciseForestParams["n_estimators"],
        criterion = preciseForestParams["criterion"],
        max_depth = preciseForestParams["max_depth"],
        max_features = preciseForestParams["max_features"])
        includedModels.append(("preciseForest",mod))
        sVs.append(displayBlock)
    else:
        sVs.append(displayHidden)
    
    newEnsemble = VotingClassifier(estimators = includedModels)
    newEnsemble.fit(X,match)
    ypredict = newEnsemble.predict(XTest)
    confusionMatrix = confusion_matrix(matchTest,ypredict)
    accuracyScore = accuracy_score(matchTest,ypredict)
    recallScore = recall_score(matchTest,ypredict)
    precisionScore = precision_score(matchTest,ypredict)

    cm = px.imshow(confusionMatrix,
    labels=dict(x="Actual", y="Predicted"),
    x=['Match Fail', 'Match Success'],
    y=['Match Fail', 'Match Success'],
    text_auto=True)
    metrics=go.Figure(go.Bar(
        x=[accuracyScore,recallScore,precisionScore], 
        y=["accuracy","recall","precision"],
        orientation='h',
        hovertext=["accuracy","recall","precision"]))
    return cm,metrics,sVs[0],sVs[1],sVs[2],sVs[3],sVs[4],sVs[5],sVs[6],sVs[7],sVs[8],sVs[9]


#sandbox callbacks

#run app
if __name__ == '__main__':
    app.run_server(debug=True)