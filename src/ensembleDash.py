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

with open("../data/plotlyDashData/collectionDictionary.json") as d:
    collectionDictionary = json.load(d)
    modelDescriptionDictionary = collectionDictionary["modelDescriptionDictionary"]
    matrixDictionary = collectionDictionary["matrixDictionary"]
    metricsTable = collectionDictionary["metricsTable"]
    significantFeaturesDictionary = collectionDictionary["significantFeaturesDictionary"]
    featureSelectOptions = collectionDictionary["featureSelectOptions"]
    selectedValue = collectionDictionary["selectedValue"]
    descriptionDictionary = collectionDictionary["descriptionDictionary"]
    selectedMatchFeatures = collectionDictionary["selectedMatchFeatures"]
    selectedMatch = pd.Series(collectionDictionary["selectedMatch"],index=selectedMatchFeatures)
    candidateFeatures = collectionDictionary["candidateFeatures"]
    partnerFeatures = collectionDictionary["partnerFeatures"]
    questionDictionary  = collectionDictionary["questionDictionary"]
    candidateProfile = collectionDictionary["candidateProfile"]
    partnerProfile = collectionDictionary["partnerProfile"]

matrixDictionaryKeys = matrixDictionary.keys()
for k in matrixDictionaryKeys:
    cm = np.array(matrixDictionary[k])
    matrixDictionary[k] = px.imshow(cm,
    labels=dict(x="Actual", y="Predicted"),
    x=['Match Fail', 'Match Success'],
    y=['Match Fail', 'Match Success'],
    text_auto=True)


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
app = Dash(__name__,suppress_callback_exceptions=True)

col12 = styles.col12
col8 = styles.col8
col6 = styles.col6
col4 = styles.col4
col3 = styles.col3
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
            ) for page in [{"name":"sandbox","path":"/sandbox"},{"name":"matchmakers","path":"/matchmakers"}]
        ],
        vertical=True,
        pills=True
    )
])

modelInfoList = []
for modelInfo in allEstimatorTuples:
    if modelInfo[0] == "Ensemble" or "knn" in modelInfo[0]:
        modInfoLayout = html.Div(style=displayBlock,id=modelInfo[0] + "Info",children=[
            html.Div(children=[
            html.Div(style=col6,children=html.H3(style=displayBlock,children=modelInfo[0])),
            html.Div(style=col6,children=html.P(children=modelDescriptionDictionary[modelInfo[0]]))
            ]),
            html.Div(children=[
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Matrix",figure=matrixDictionary[modelInfo[0]])
                ]),
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Metrics",
                    figure = go.Figure(go.Bar(
                        x=list(metricsTable[modelInfo[0]].values()), 
                        y=list(metricsTable[modelInfo[0]].keys()),
                        orientation='h',
                        hovertext=list(metricsTable[modelInfo[0]].values()))))
                ])
            ])
        ])
    else:
        modInfoLayout = html.Div(style=displayBlock,id=modelInfo[0] + "Info",children=[
            html.Div(children=[
            html.Div(style=col6,children=html.H3(style=displayBlock,children=modelInfo[0])),
            html.Div(style=col6,children=html.P(children=modelDescriptionDictionary[modelInfo[0]]))
            ]),
            html.Div(children=[
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Matrix",figure=matrixDictionary[modelInfo[0]])
                ]),
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Metrics",
                    figure = go.Figure(go.Bar(
                        x=list(metricsTable[modelInfo[0]].values()), 
                        y=list(metricsTable[modelInfo[0]].keys()),
                        orientation='h',
                        hovertext=list(metricsTable[modelInfo[0]].values()))))
                ])
            ]),
            html.Div(children=[
                html.H4("Top 10 deciding features:"),
                html.Ol(children=[
                    html.Li(children=feat) for feat in (significantFeaturesDictionary[modelInfo[0]])
                ])
            ])
        ])
    modelInfoList.append(dcc.Loading(children=modInfoLayout))

matchmakersLayout = html.Div(style=col12,children=modelInfoList)

featureSelect = dcc.Dropdown(
    id="featureSelect",
    options=featureSelectOptions,
    value=[]
)

featureNumber = html.Div(id='featureNumber',style=displayBlock,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Input(id="featureNumberInput",type="number")
])

featureDropdown = html.Div(id='featureNumber',style=displayBlock,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Dropdown(id='featureDropdownInput')
])

sandboxLayout = html.Div(style=col12,children=[
    html.Div(children=[
        html.Div(style=col4,children=[
            html.H3(style=displayBlock,children="What do you want to examine"),
            featureSelect
        ]),
        html.Div(style=col8,children=[
            html.H3(style=col12,children="Calculated Values: Edit profile traits to change these"),
            html.Div(style=col6,children=[
                html.Div(style=col12,title=descriptionDictionary["partnerDistance"],children=[
                    html.Span(children="Partner Distance: "),
                    html.Span(id="partnerDistance",children=str(round(selectedMatch["partnerDistance"]))),
                    html.Span(children="miles")
                ]),
                html.Div(style=col12,title=descriptionDictionary["samerace"],children=[
                    html.Span(children="Same race?: "),
                    html.Span(id="samerace",children = "yes" if (selectedMatch["samerace"]==1) else "no")
                ])
            ]),
            html.Div(style=col6,children=[
                html.Div(style=col6,id="sharedInterestsValue",children=[
                    html.Span(children="Shared Interest Correlation:"),
                    html.Span(id="int_corr",children=selectedMatch["int_corr"])
                ]),
                html.Div(style=col6,children=dcc.Graph(style=col12,id="sharedInterestsGraph"))
            ])
        ])
        
    ]),
    html.Div(children=[
        html.Div(style=col3,children=[
            html.H3(style=displayBlock,children="candidate features"),
            html.Div(children=[
                html.Div(id=str(col)+"Display",title=descriptionDictionary[col],children = [
                    html.Span(children=f"{col}: "),
                    html.Span(id=str(col)+"Value",children=candidateProfile[col])
                ]) 
                for col in candidateFeatures
            ])
        ]),
        html.Div(style=col6,children=[
            html.Div(children=dcc.Graph(style=col12,id="predictProbaGraph")),
            html.Div(children=dcc.Graph(style=col12,id="diversityGraph"))
        ]),
        html.Div(style=col3,children=[
            html.H3(style=displayBlock,children="partner features"),
            html.Div(children=[
                html.Div(id=str(col)+"Edit",title=descriptionDictionary[col],children = [
                    html.Span(children=f"{col}: "),
                    html.Span(id=str(col)+"Value",children=partnerProfile[col])
                ]) 
                for col in partnerFeatures
            ])
        ])
    ])
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
        html.Div(id="content")
    ]) 
])

@app.callback(
    [dash.dependencies.Output('pagetitle','children'),
    dash.dependencies.Output('content','children')],
    Input('url', 'pathname'))
def updatePage(pathname):
    print(pathname)
    if "sandbox" in pathname.lower() or pathname.lower() == "/":
        return "Sandbox",sandboxLayout
    return "Matchmakers",matchmakersLayout

#modelSection callback
@app.callback(
    [dash.dependencies.Output('EnsembleMatrix', 'figure'),
    dash.dependencies.Output('EnsembleMetrics', 'figure'),
    dash.dependencies.Output('EnsembleInfo','style')],
    [dash.dependencies.Input('modelSelection', 'value')], prevent_initial_call=True)
def updateEnsembleInfo(models):
    includedModels = []
    styleValue = displayBlock
    if len(models)==0:
        return px.imshow(np.zeros((2,2))),go.Figure(go.Bar(dict())),displayHidden
    if len(models)==1:
        return px.imshow(np.zeros((2,2))),go.Figure(go.Bar(dict())),displayHidden

    if ("logModel" in models):
        premod = lm.LogisticRegression(max_iter=1e9)
        mod = make_pipeline(StandardScaler(), premod)
        includedModels.append(("logModel",mod))
    if ("knn5" in models):
        mod = knn(n_neighbors=5)
        includedModels.append(("knn5",mod))
    if ("knnsqrtn" in models):
        mod = knn(n_neighbors=sqrtn)
        includedModels.append(("knnsqrtn",mod))
    if ("gradientdeci" in models):
        mod = grad(learning_rate=0.1)
        includedModels.append(("gradientdeci",mod))
    if ("gradientdeka" in models):
        mod = grad(learning_rate=10)
        includedModels.append(("gradientdeka",mod))
    if ("recallTree" in models):
        mod = tree(criterion = recallTreeParams["criterion"],
        max_depth = recallTreeParams["max_depth"],
        max_features = recallTreeParams["max_features"])
        includedModels.append(("recallTree",mod))
    if ("preciseTree" in models):
        mod = tree(criterion = preciseTreeParams["criterion"],
        max_depth = preciseTreeParams["max_depth"],
        max_features = preciseTreeParams["max_features"])
        includedModels.append(("preciseTree",mod))
    if ("recallForest" in models):
        mod = rf(n_estimators = recallForestParams["n_estimators"],
        criterion = recallForestParams["criterion"],
        max_depth = recallForestParams["max_depth"],
        max_features = recallForestParams["max_features"])
        includedModels.append(("recallForest",mod))
    if ("preciseForest" in models):
        mod = rf(n_estimators = preciseForestParams["n_estimators"],
        criterion = preciseForestParams["criterion"],
        max_depth = preciseForestParams["max_depth"],
        max_features = preciseForestParams["max_features"])
        includedModels.append(("preciseForest",mod))
    
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
    return cm,metrics,displayBlock

@dash.callback(
    Output('logModelInfo','style'),
    Input('modelSelection', 'value'))
def updatelogModelInfo(models):
    hideValue = displayHidden
    if "logModel" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('knn5Info','style'),
    Input('modelSelection', 'value'))
def updateknn5Info(models):
    hideValue = displayHidden
    if "knn5" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('knnsqrtnInfo','style'),
    Input('modelSelection', 'value'))
def updateknnsqrtnInfo(models):
    hideValue = displayHidden
    if "knnsqrtn" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('gradientdeciInfo','style'),
    Input('modelSelection', 'value'))
def updategradientdeciInfo(models):
    hideValue = displayHidden
    if "gradientdeci" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('gradientdekaInfo','style'),
    Input('modelSelection', 'value'))
def updategradientdekaInfo(models):
    hideValue = displayHidden
    if "gradientdeka" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('recallTreeInfo','style'),
    Input('modelSelection', 'value'))
def updaterecallTreeInfo(models):
    hideValue = displayHidden
    if "recallTree" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('preciseTreeInfo','style'),
    Input('modelSelection', 'value'))
def updatepreciseTreeInfo(models):
    hideValue = displayHidden
    if "preciseTree" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('recallForestInfo','style'),
    Input('modelSelection', 'value'))
def updaterecallForestInfo(models):
    hideValue = displayHidden
    if "recallForest" in models:
        hideValue = displayBlock
    return hideValue

@dash.callback(
    Output('preciseForestInfo','style'),
    Input('modelSelection', 'value'))
def updatepreciseForestInfo(models):
    hideValue = displayHidden
    if "preciseForest" in models:
        hideValue = displayBlock
    return hideValue


#sandbox callbacks

#run app
if __name__ == '__main__':
    app.run_server(debug=True)