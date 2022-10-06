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
import preprocess

originalEstimtatorTuples = preprocess.originalEstimtatorTuples
sqrtn = preprocess.sqrtn
recallTreeParams = preprocess.recallTreeParams
preciseTreeParams = preprocess.preciseTreeParams
recallForestParams = preprocess.recallForestParams
preciseForestParams = preprocess.preciseForestParams
X = preprocess.X
match = preprocess.match
XTest = preprocess.XTest
matchTest = preprocess.matchTest
# Dash code
app = Dash(__name__, use_pages=True)

middleAndCenter = styles.middleAndCenter
selected = styles.selected
unselected = styles.unselected
fitContent = styles.fitContent
sidebarstyle = styles.SIDEBAR_STYLE
contentstyle = styles.CONTENT_STYLE

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
    html.Div(style=contentstyle,children=[
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

#modelSection callback
@dash.callback(
    Output('EnsembleMatrix', 'figure'),
    Output('EnsembleMetrics', 'figure'),
    Output('EnsembleInfo','hidden'),
    Input('modelSelection', 'value'))
def updateEnsembleInfo(models):
    includedModels = []
    hide="hidden"
    if len(models)==0:
        models = [modelTuple[0] for modelTuple in originalEstimtatorTuples]
    else:
        hide=""
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
    labels=dict(x="Actual", y="Predicted", color="Productivity"),
                x=['Match Fail', 'Match Success'],
                y=['Match Fail', 'Match Success'],
    text_auto=True)
    metrics=go.Figure(go.Bar(
        x=[accuracyScore,recallScore,precisionScore], 
        y=["accuracy","recall","precision"],
        orientation='h',
        hovertext=["accuracy","recall","precision"]))

    return cm,metrics

@dash.callback(
    Output('logModelInfo','hidden'),
    Input('modelSelection', 'value'))
def updatelogModelInfo(models):
    hideValue = "hidden"
    if "logModel" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('knn5Info','hidden'),
    Input('modelSelection', 'value'))
def updateknn5Info(models):
    hideValue = "hidden"
    if "knn5" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('knnsqrtnInfo','hidden'),
    Input('modelSelection', 'value'))
def updateknnsqrtnInfo(models):
    hideValue = "hidden"
    if "knnsqrtn" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('gradientdeciInfo','hidden'),
    Input('modelSelection', 'value'))
def updategradientdeciInfo(models):
    hideValue = "hidden"
    if "gradientdeci" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('gradientdekaInfo','hidden'),
    Input('modelSelection', 'value'))
def updategradientdekaInfo(models):
    hideValue = "hidden"
    if "gradientdeka" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('recallTreeInfo','hidden'),
    Input('modelSelection', 'value'))
def updaterecallTreeInfo(models):
    hideValue = "hidden"
    if "recallTree" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('preciseTreeInfo','hidden'),
    Input('modelSelection', 'value'))
def updatepreciseTreeInfo(models):
    hideValue = "hidden"
    if "preciseTree" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('recallForestInfo','hidden'),
    Input('modelSelection', 'value'))
def updaterecallForestInfo(models):
    hideValue = "hidden"
    if "recallForest" in models:
        hideValue = ""
    return hideValue

@dash.callback(
    Output('preciseForestInfo','hidden'),
    Input('modelSelection', 'value'))
def updatepreciseForestInfo(models):
    hideValue = "hidden"
    if "preciseForest" in models:
        hideValue = ""
    return hideValue

#sandbox callbacks

#run app
if __name__ == '__main__':
    app.run_server(debug=True)