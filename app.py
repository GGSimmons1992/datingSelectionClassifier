import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr
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
import src.styles as styles
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

with open("data/plotlyDashData/collectionDictionary.json") as d:
    collectionDictionary = json.load(d)
    modelDescriptionDictionary = collectionDictionary["modelDescriptionDictionary"]
    matrixDictionary = collectionDictionary["matrixDictionary"]
    metricsTable = collectionDictionary["metricsTable"]
    significantFeaturesDictionary = collectionDictionary["significantFeaturesDictionary"]
    featureSelectOptions = collectionDictionary["featureSelectOptions"]

with open("data/plotlyDashData/dummyDictionary.json") as d:
    dummyDictionary = json.load(d)

with open("data/dummyValueDictionary.json") as d:
    dummyValueDictionary = json.load(d)

matrixDictionaryKeys = matrixDictionary.keys()
for k in matrixDictionaryKeys:
    cm = np.array(matrixDictionary[k])
    matrixDictionary[k] = px.imshow(cm,
    labels=dict(x="Actual", y="Predicted"),
    x=['Match Fail', 'Match Success'],
    y=['Match Fail', 'Match Success'],
    text_auto=True)


datingTrain = pd.read_csv('data/plotlyDashData/datingTrain.csv')
datingTest = pd.read_csv('data/plotlyDashData/datingTest.csv')

match = datingTrain["match"]
X = datingTrain.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])
matchTest = datingTest["match"]
XTest = datingTest.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])

with open('data/processedData/treeParams.json') as d:
    treeParams = json.load(d)
    preciseTreeParams = treeParams["preciseTreeParams"]
    recallTreeParams = treeParams["recallTreeParams"]
with open('data/processedData/forestParams.json') as d:
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

def generateIncludedModels(models):
    includedModels = []
    
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
    
    return includedModels

def createDFFromDictionary(dictionary):
    return pd.DataFrame(dictionary,columns=list(dictionary.keys()))

def createStatisticsFromDummies(newEnsemble,featureParam,fullX,fully):
    newEnsemble.train(fullX,fully)
    dummyCols = dummyDictionary[featureParam]
    yPredictFull = newEnsemble.predict_proba(fullX)[1]
    correlationDictionary = dict()
    correlationDictionary["label"] = []
    correlationDictionary["spearman r value"] = []
    correlationDictionary["p"] = []
    correlationDictionary["color"] = []

    for dummyCol in dummyCols:
        dummyLabel = dummyValueDictionary[dummyCol]
        corr,p = spearmanr(list(fullX[dummyCol]),list(yPredictFull))
        significanceColor = "green" if p < 0.05 else "red"
        correlationDictionary["label"].append(dummyLabel)
        correlationDictionary["spearman r value"].append(corr)
        correlationDictionary["p"].append(p)
        correlationDictionary["color"].append(significanceColor)

    resultsDF = createDFFromDictionary(correlationDictionary)

    fig = go.bar(resultsDF)

    return fig

def createCorrelationsFromDummies(newEnsemble,featureParam,fullX,fully):
    newEnsemble.train(fullX,fully)
    dummyCols = dummyDictionary[featureParam]
    yPredictFull = newEnsemble.predict_proba(fullX)[1]
    correlationDictionary = dict()
    correlationDictionary["label"] = []
    correlationDictionary["spearman r value"] = []
    correlationDictionary["p"] = []
    correlationDictionary["color"] = []

    for dummyCol in dummyCols:
        dummyLabel = dummyValueDictionary[dummyCol]
        corr,p = spearmanr(list(fullX[dummyCol]),list(yPredictFull))
        significanceColor = "green" if p < 0.05 else "red"
        correlationDictionary["label"].append(dummyLabel)
        correlationDictionary["spearman r value"].append(corr)
        correlationDictionary["p"].append(p)
        correlationDictionary["color"].append(significanceColor)

    resultsDF = createDFFromDictionary(correlationDictionary)

    fig = go.bar(resultsDF)

def createStatisticsFromRange(allModels,featureParam,matchProfile):
    resultsDF = pd.DataFrame()
    return resultsDF

def createCorrelationsFromRange(allModels,featureParam,matchProfile):
    resultsDF = pd.DataFrame()
    return resultsDF

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
            ) for page in [{"name":"Feature Analysis","path":"/featureanalysis"},{"name":"matchmakers","path":"/matchmakers"}]
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

featureAnalysisTemplates = [
    html.Div(id=genderType+"Info",children=[
        html.H3(children = f"If candidate was {genderType}" if genderType != "overall" else "For both genders"),
        html.div(children=[
            html.div(style=col6,children=html.div(style=col12,children=dcc.Graph(id=genderType+"Diversity"))),
            html.div(style=col6,children=html.div(style=col12,children=dcc.Graph(id=genderType+"Statistics"))),
        ]),
        html.div(chilren=dcc.Graph(id=genderType+"Correlations"))
    ]) for genderType in ["male","female","overall"]]

featureAnalysisLayout = html.Div(children=featureAnalysisTemplates)

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
    if "featureanalysis" in pathname.lower() or pathname.lower() == "/":
        return "Feature Analysis",featureAnalysisLayout
    return "Matchmakers",matchmakersLayout

#modelSection callback
@app.callback(
    [dash.dependencies.Output('EnsembleMatrix', 'figure'),
    dash.dependencies.Output('EnsembleMetrics', 'figure'),
    dash.dependencies.Output('EnsembleInfo','style')],
    [dash.dependencies.Input('modelSelection', 'value')], prevent_initial_call=True)
def updateEnsembleInfo(models):
    if len(models)==0:
        return px.imshow(np.zeros((2,2))),go.Figure(go.Bar(dict())),displayHidden
    if len(models)==1:
        return px.imshow(np.zeros((2,2))),go.Figure(go.Bar(dict())),displayHidden

    includedModels = generateIncludedModels(models)
    
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

# @app.callback(
#     [dash.dependencies.Output('predictProbaGraph','figure')],
#     [dash.dependencies.Input()]
# )
# def fitProba(models,featureParam,matchProfile):
#     includedModels = generateIncludedModels
#     newEnsemble = VotingClassifier(estimators = includedModels)
#     allModels = [("Ensemble",newEnsemble)] + includedModels

#     if len(list(set(datingTest[featureParam]))) < 30:
#         resultsDF = createResultsDFFromDummies(newEnsemble,featureParam,matchProfile)
#         fig = go.Figure(go.Bar(resultsDF,
#         x=featureParam, 
#         y="predictedSuccess",
#         orientation='h',
#         hovertext=["accuracy","recall","precision"]))
#     else:
#         resultsDF = createResultsDFFromRange(allModels,featureParam,matchProfile)
#         fig = px.line(resultsDF,x=featureParam,y='predictedSuccess',color = 'modelName',
#         labels= {featureParam:descriptionDictionary[featureParam],"modelName":"Model Name"},
#         title=f"Predicted Success vs {descriptionDictionary[featureParam]}")
        
#     return fig

# @app.callback(
#     [dash.dependencies.Output('diversityGraph','figure')],
#     [dash.dependencies.Input()]
# )
# def fitHist(featureParam):
#     pass





#run app
if __name__ == '__main__':
    app.run_server(debug=True)