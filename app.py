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

originalDummyDictionaryKeys = list(dummyDictionary.keys())
for originalKey in originalDummyDictionaryKeys:
    dummyDictionary[originalKey+"_o"] = [str(col)+"_o" for col in dummyDictionary[originalKey]]

with open("dummyValueDictionary.json") as d:
    dummyValueDictionary = json.load(d)

originalDummyValueDictionaryKeys = list(dummyValueDictionary.keys())
for originalKey in originalDummyValueDictionaryKeys:
    dummyValueDictionary[originalKey+"_o"] = dummyValueDictionary[originalKey]

with open("descriptionDictionary.json") as d:
    descriptionDictionary = json.load(d)

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
datingFull = pd.read_csv('data/plotlyDashData/datingFull.csv')

match = datingTrain["match"]
X = datingTrain.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])
matchTest = datingTest["match"]
XTest = datingTest.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])

matchFull = datingFull["match"]
datingMale = datingFull[datingFull["gender"]==1].drop("match",axis=1)
datingFemale = datingFull[datingFull["gender"]==0].drop("match",axis=1)
datingFull = datingFull.drop("match",axis=1)

with open('data/plotlyDashData/treeParams.json') as d:
    treeParams = json.load(d)
    preciseTreeParams = treeParams["preciseTreeParams"]
    recallTreeParams = treeParams["recallTreeParams"]
with open('data/plotlyDashData/forestParams.json') as d:
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


for estimatorTuple in originalEstimtatorTuples:
    (estimatorTuple[1]).fit(X,match)
ensembleVote.fit(X,match)

allEstimatorTuples = [("Ensemble",ensembleVote)] + originalEstimtatorTuples

defaultBar = px.bar(pd.DataFrame({"blank":[]},columns=["blank"]))

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

def createDistributionFromDummies(featureParam,fullX,figTitle):
    dummyCols = dummyDictionary[featureParam]
    labels = [dummyValueDictionary[dummyCol] for dummyCol in dummyCols]
    
    counts = [sum(fullX[col]) for col in dummyCols]
    yLabel = f"{featureParam} Values"
    resultsDF = pd.DataFrame({
        "Counts":counts,
        yLabel: labels
    },columns = ["Counts",yLabel])

    fig = px.bar(resultsDF, x="Counts",y=yLabel,
    orientation="h",labels={"Counts":"Counts",yLabel:yLabel},
    title=figTitle)

    return fig

def createStatisticsFromDummies(newEnsemble,featureParam,fullX,figTitle):
    newEnsemble.fit(X,match)
    
    dummyCols = dummyDictionary[featureParam]
    statisticsDictionary = dict()
    statisticsDictionary["label"] = []
    statisticsDictionary["mean"] = []
    statisticsDictionary["standard error"] = []
    for dummyCol in dummyCols:
        selectedRows = fullX[fullX[dummyCol]==1]
        if selectedRows.shape[0] == 0:
            pass
        else:
            selectedX = selectedRows
            predicty = (newEnsemble.predict_proba(selectedX)[:,1])
            yMean = np.mean(predicty)
            ySE = np.std(predicty)/np.sqrt(len(predicty))
            statisticsDictionary["label"].append(dummyValueDictionary[dummyCol])
            statisticsDictionary["mean"].append(yMean)
            statisticsDictionary["standard error"].append(ySE)
    
    resultsDF = createDFFromDictionary(statisticsDictionary)

    fig = px.bar(resultsDF, x="mean",y="label",error_x="standard error",
    orientation="h",labels={"mean":"mean predicted probability","label":featureParam+" value"},
    title=figTitle)

    return fig

def createCorrelationsFromDummies(newEnsemble,featureParam,fullX,figTitle):
    newEnsemble.fit(X,match)
    dummyCols = dummyDictionary[featureParam]
    yPredictFull = np.array(newEnsemble.predict_proba(fullX)[:,1]).reshape(-1,).tolist()
    correlationDictionary = dict()
    correlationDictionary["label"] = []
    correlationDictionary["spearman r value"] = []
    correlationDictionary["significance"] = []

    for dummyCol in dummyCols:
        dummyLabel = dummyValueDictionary[dummyCol]
        corr,p = spearmanr(np.array(fullX[dummyCol]).reshape(-1,).tolist(),list(yPredictFull))
        significance = "Significant" if p < 0.05 else "Not Significant"
        correlationDictionary["label"].append(dummyLabel + f" p={np.round(p,2)}")
        correlationDictionary["spearman r value"].append(corr)
        correlationDictionary["significance"].append(significance)

    resultsDF = createDFFromDictionary(correlationDictionary)

    fig = px.bar(resultsDF, x="spearman r value",y="label",color="significance",color_discrete_map={
        'Not Significant': 'red',
        'Significant': 'green'
    },
    orientation="h",labels={"spearman r value":"Spearman R Correlation Value","label":featureParam+" Value and P-Value"},
    title=figTitle)

    return fig

def createDistributionFromSamerace():
    
    resultsDF = pd.DataFrame({
        "Counts": [datingFull[datingFull["samerace"] == val].shape[0] for val in [0,1]],
        "Pairing": ["Different Race", "Same Race"]
    },columns=["Counts","Pairing"])
    
    fig = px.bar(resultsDF,x="Counts",y="Pairing",
    title="Number of Entries that are Same/Different Race")

    return fig 

def createStatisticsFromSamerace(newEnsemble):
    newEnsemble.fit(X,match)
    sameraceDF = datingFull[datingFull["samerace"]==1]
    differentDF = datingFull[datingFull["samerace"]==0]

    statisticsDictionary = dict()
    statisticsDictionary["predicted probability"] = []
    statisticsDictionary["label"] = ["Different Race","Same Race"]
    statisticsDictionary["standard error"] = []

    for df in [differentDF,sameraceDF]:
        probabilities = list(newEnsemble.predict_proba(df)[:,1])
        statisticsDictionary["predicted probability"].append(np.mean(probabilities))
        statisticsDictionary["standard error"].append(np.std(probabilities)/np.sqrt(df.shape[0]))

    resultsDF = createDFFromDictionary(statisticsDictionary) 

    fig = px.bar(resultsDF, x="predicted probability",y="label",error_x="standard error",
    orientation="h",
    title="Statistics on Same/Different Race")

    return fig

def createCorrelationsFromSamerace(newEnsemble):
    actualMatch = np.array(matchFull).reshape(-1,).tolist()
    newEnsemble.fit(X,match)
    sameraceValue = np.array(datingFull["samerace"]).reshape(-1,).tolist()
    predicty = np.array(newEnsemble.predict_proba(datingFull)[:,1]).reshape(-1,).tolist()
    corr,p = spearmanr(sameraceValue,predicty)
    significanceValue = "Signficant" if p < 0.05 else "Not Significant"

    resultsDF = pd.DataFrame({
        "Same Race Value" : sameraceValue,
        "Predicted Probability" : predicty,
        "Actual Match Result" : actualMatch
    },columns = ["Same Race Value","Predicted Probability","Actual Match Result"])

    fig = px.scatter(resultsDF,x="Same Race Value",y="Predicted Probability",color="Actual Match Result",
    title=f"Predicted Probability vs Same Race Value R={np.round(corr,2)} p={np.round(p,2)} {significanceValue}")

    return fig

def createDistributionFromRange(featureParam,fullX):

    featureData = list(fullX[featureParam])

    fig = ff.create_distplot([featureData],[f"{featureParam} Distribution"])

    fig.update_layout(title=f"{featureParam} Distribution")

    return fig

def createStatisticsFromRange(newEnsemble,selectedModels,featureParam,fullX):
    newEnsemble.fit(X,match)
    fullXSorted = fullX.copy().sort_values(by=featureParam)
    allModels = [("Ensemble",newEnsemble)] + [modelTuple for modelTuple in originalEstimtatorTuples if modelTuple[0] in selectedModels]
    
    statisticsDictionary = dict()
    statisticsDictionary[featureParam] = np.array(fullXSorted[featureParam]).reshape(-1,).tolist()

    for selectedModel in allModels:
        statisticsDictionary[selectedModel[0]] = np.array(selectedModel[1].predict_proba(fullXSorted)[:,1]).reshape(-1,).tolist()

    resultsDF = createDFFromDictionary(statisticsDictionary)

    fig = px.scatter(resultsDF, x=featureParam, y="Ensemble",labels={"Ensemble":"Predicted Probability"},
     title=f'Predicted probability of match based on {descriptionDictionary[featureParam]}')

    return fig

def createCorrelationsFromRange(newEnsemble,selectedModels,featureParam,fullX,figTitle):
    newEnsemble.fit(X,match)
    allModels = [("Ensemble",newEnsemble)] + [modelTuple for modelTuple in originalEstimtatorTuples if modelTuple[0] in selectedModels]

    correlationDictionary = dict()
    correlationDictionary["model"] = []
    correlationDictionary["spearman r value"] = []
    correlationDictionary["significance"] = []
    featureValues = np.array(fullX[featureParam]).reshape(-1).tolist()
    for modelTuple in allModels:
        predicty = np.array(modelTuple[1].predict_proba(fullX)[:,1]).reshape(-1).tolist()
        corr,p = spearmanr(featureValues,predicty)
        significance = "Significant" if p < 0.05 else "Not Significant"
        correlationDictionary["model"].append(modelTuple[0] + f" p={np.round(p,2)}")
        correlationDictionary["spearman r value"].append(corr)
        correlationDictionary["significance"].append(significance)

    resultsDF = createDFFromDictionary(correlationDictionary)

    fig = px.bar(resultsDF, x="spearman r value",y="model",color="significance",color_discrete_map={
        'Not Significant': 'red',
        'Significant': 'green'
    },
    orientation="h",labels={"spearman r value":"Spearman R Correlation Value","model":"Model and P-Value"},
    title=figTitle)

    return fig

# Dash code
app = Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            ) for page in [{"name":"Home","path":"/"},
            {"name":"Matchmakers","path":"/matchmakers"},
            {"name":"Feature Analysis","path":"/featureanalysis"}]
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
            ]),
            html.Hr()
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
            ]),
            html.Hr()
        ])
    modelInfoList.append(dcc.Loading(children=modInfoLayout))

matchmakersLayout = html.Div(style=col12,children=modelInfoList)

featureAnalysisTemplates = [
    dcc.Loading(
        html.Div(style=displayHidden,id=genderType+"Info",children=[
            html.H4(children = f"If candidate was {genderType}" if genderType != "overall" else "For both genders"),
            html.Div(children=[
                html.Div(style=col6,children=html.Div(style=col12,children=dcc.Loading(dcc.Graph(id=genderType+"Diversity")))),
                html.Div(style=col6,children=html.Div(style=col12,children=dcc.Loading(dcc.Graph(id=genderType+"Statistics")))),
            ]),
            html.Div(children=dcc.Loading(dcc.Graph(id=genderType+"Correlations"))),
            html.Hr()
        ])
    ) for genderType in ["male","female","overall"]]


featureSelect = dcc.Dropdown(
    id="featureSelect",
    options=featureSelectOptions,
    value=""
)

featureAnalysisLayout = html.Div(children= [
    html.Div(children=[
        html.H3(children="What do you want to analyze?"),
        featureSelect
    ])
] + featureAnalysisTemplates)

app.layout = html.Div(children= [
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(style=contentstyle,children=[
        html.H1(children='Blind Dating Ensemble Classifier Dashboard',style={"background-color":"dodgerblue"}),
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
    if pathname.lower() == "/":
        return "Home", html.H3("Welcome to the Blind Dating Ensemble Classifier Dashboard")
    elif "featureanalysis" in pathname.lower():
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
        return px.imshow(np.zeros((2,2))),go.Figure(defaultBar),displayHidden
    if len(models)==1:
        return px.imshow(np.zeros((2,2))),go.Figure(defaultBar),displayHidden

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

#Analysis callbacks
@app.callback(
    Output("maleInfo","style"),
    Output("femaleInfo","style"),
    Output("overallInfo","style"),
    Input('featureSelect',"value"),
        prevent_initial_call=True)
def updateStyles(featureParam):
    if featureParam == "":
        return displayHidden,displayHidden,displayHidden
    elif featureParam == "samerace":
        return displayHidden,displayHidden,displayBlock
    else:
        return displayBlock,displayBlock,displayBlock

@app.callback(
    Output("maleDiversity","figure"),
    Output("femaleDiversity","figure"),
    Output("overallDiversity","figure"),
    Input('featureSelect',"value"),
        prevent_initial_call=True)
def updateDistributions(featureParam):
    if featureParam == "":
        return defaultBar,defaultBar,defaultBar
    elif featureParam == "samerace":
        return defaultBar,defaultBar,createDistributionFromSamerace()
    else:
        if featureParam in dummyDictionary.keys():
            m = createDistributionFromDummies(featureParam,datingMale,f"Distribution Amongst {descriptionDictionary[featureParam]} Values")
            f = createDistributionFromDummies(featureParam,datingFemale,f"Distribution Amongst {descriptionDictionary[featureParam]} Values")
            o = createDistributionFromDummies(featureParam,datingFull,f"Distribution Amongst {descriptionDictionary[featureParam]} Values")
        else:
            m = createDistributionFromRange(featureParam,datingMale)
            f = createDistributionFromRange(featureParam,datingFemale)
            o = createDistributionFromRange(featureParam,datingFull)
        return m,f,o

@app.callback(
    Output("maleStatistics","figure"),
    Output("femaleStatistics","figure"),
    Output("overallStatistics","figure"),
    Input('modelSelection','value'),
    Input('featureSelect',"value"),
        prevent_initial_call=True)
def updateStatistics(models,featureParam):
    if featureParam == "":
        return defaultBar,defaultBar,defaultBar
    selectedModels = generateIncludedModels(models)
    newEnsemble = VotingClassifier(estimators=selectedModels,voting="soft")
    if featureParam == "samerace":
        return defaultBar,defaultBar,createStatisticsFromSamerace(newEnsemble)
    else:
        if featureParam in dummyDictionary.keys():
            m = createStatisticsFromDummies(newEnsemble,featureParam,datingMale,f"{descriptionDictionary[featureParam]} Statistics")
            f = createStatisticsFromDummies(newEnsemble,featureParam,datingFemale,f"{descriptionDictionary[featureParam]} Statistics")
            o = createStatisticsFromDummies(newEnsemble,featureParam,datingFull,f"{descriptionDictionary[featureParam]} Statistics")
        else:
            m = createStatisticsFromRange(newEnsemble,models,featureParam,datingMale)
            f = createStatisticsFromRange(newEnsemble,models,featureParam,datingFemale)
            o = createStatisticsFromRange(newEnsemble,models,featureParam,datingFull)
        return m,f,o

@app.callback(
    Output("maleCorrelations","figure"),
    Output("femaleCorrelations","figure"),
    Output("overallCorrelations","figure"),
    Input('modelSelection','value'),
    Input('featureSelect',"value"),
        prevent_initial_call=True)
def updateCorrelations(models,featureParam):
    if featureParam == "":
        return defaultBar,defaultBar,defaultBar
    selectedModels = generateIncludedModels(models)
    newEnsemble = VotingClassifier(estimators=selectedModels,voting="soft")
    if featureParam == "samerace":
        return defaultBar,defaultBar,createCorrelationsFromSamerace(newEnsemble)
    else:
        if featureParam in dummyDictionary.keys():
            m = createCorrelationsFromDummies(newEnsemble,featureParam,datingMale,f"{descriptionDictionary[featureParam]} Correlations")
            f = createCorrelationsFromDummies(newEnsemble,featureParam,datingFemale,f"{descriptionDictionary[featureParam]} Correlations")
            o = createCorrelationsFromDummies(newEnsemble,featureParam,datingFull,f"{descriptionDictionary[featureParam]} Correlations")
        else:
            m = createCorrelationsFromRange(newEnsemble,models,featureParam,datingMale,f"{descriptionDictionary[featureParam]} Correlations")
            f = createCorrelationsFromRange(newEnsemble,models,featureParam,datingFemale,f"{descriptionDictionary[featureParam]} Correlations")
            o = createCorrelationsFromRange(newEnsemble,models,featureParam,datingFull,f"{descriptionDictionary[featureParam]} Correlations")
        return m,f,o

#run app
if __name__ == '__main__':
    app.run_server(debug=True)