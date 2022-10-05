import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import styles
import ensembleDash

significantFeaturesDictionary = dict()

X = ensembleDash.X
match = ensembleDash.match
XTest = ensembleDash.XTest
matchTest = ensembleDash.matchTest
allEstimatorTuples = ensembleDash.allEstimatorTuples
descriptionDictionary = ensembleDash.descriptionDictionary
preciseTreeParams = ensembleDash.preciseTreeParams
recallTreeParams = ensembleDash.recallTreeParams
preciseForestParams = ensembleDash.preciseForestParams
recallForestParams = ensembleDash.recallForestParams

XColumns = list(X.columns)

matchLabels = ["non-match","match"]

modelNames =  [estimatorTuple[0] for estimatorTuple in allEstimatorTuples]
accuracyScores = []
recallScores = []
precisionScores = []

for modelTuple in allEstimatorTuples:
    ypredict = (modelTuple[1]).predict(XTest)
    confusionMatrix = confusion_matrix(matchTest,ypredict)
    accuracyScore = accuracy_score(matchTest,ypredict)
    recallScore = recall_score(matchTest,ypredict)
    precisionScore = precision_score(matchTest,ypredict)
    if (recallScore==0.0 or precisionScore==0.0):
        tn, fp, fn, tp = confusionMatrix.ravel()
        recallScore = tp/(tp+fn)
        precisionScore = tp/(tp+fp)
    accuracyScores.append(accuracyScore)
    recallScores.append(recallScore)
    precisionScores.append(precisionScore)

    if "log" in modelTuple[0]:
        coefficients = [coef for coef in (modelTuple[1]).named_steps['logisticregression'].coef_.reshape(-1,)]
        absCoefficients = [absCoef for absCoef in np.abs(np.array(coefficients))]

        logImportances = pd.DataFrame({
            "feature": XColumns,
            "featureImportance": coefficients,
            "absCoefficients": absCoefficients},columns = ["feature","featureImportance","absCoefficients"])
        logImportancesSorted = logImportances.sort_values(by="absCoefficients", ascending=False).head(10)
        topFeatures = logImportancesSorted["feature"].to_list()
        featureDescriptions = [descriptionDictionary[feat] for feat in topFeatures]
        logImportancesSorted["description"] = pd.Series(featureDescriptions)
        logImportancesSorted["rank"] = pd.Series(list(range(1,11)))
        significantFeaturesDictionary[modelTuple[0]] = logImportancesSorted[["rank","feature","description"]]
    elif "knn" not in modelTuple[0]:
        importance = (modelTuple[1]).feature_importances_
        featureImportance = pd.DataFrame({
            "feature": XColumns,
            "featureImportance": importance
            },columns = ["feature","featureImportance"])
        featureImportancesSorted = featureImportance.sort_values(by="featureImportance", ascending=False).head(10)
        topFeatures = featureImportancesSorted["feature"].to_list()
        featureDescriptions = [descriptionDictionary[feat] for feat in topFeatures]
        featureImportancesSorted["description"] = pd.Series(featureDescriptions)
        featureImportancesSorted["rank"] = pd.Series(list(range(1,11)))
        significantFeaturesDictionary[modelTuple[0]] = featureImportancesSorted[["rank","feature","description"]]

metricsTable = pd.DataFrame({
    "name":modelNames,
    "accuracy":accuracyScores,
    "recall":recallScores,
    "precision":precisionScores
},columns=["name","accuracy","recall","precision"])

modelDescriptionDictionary = {
    "Ensemble":"ensembleVote classifier using selected parameters above",
    "logModel":"logarithmic regression with a standard scaler",
    "knn5":"k-neighbors with k=5",
    "knnsqrtn":"k-neighbors with k=(n_features)",
    "gradientdeci":"gradient boosting classifer with learning rate = 0.1",
    "gradientdeka":"gradient boosting classifer with learning rate = 10",
    "preciseTree":"decision tree trained for best precision" + str(preciseTreeParams),
    "recallTree":"decision tree trained for best recall" + str(recallTreeParams),
    "preciseForest":"random forest trained for best precision" + str(preciseForestParams),
    "recallForest":"random forest trained for best recall" + str(recallForestParams)
}

Dash.register_page(__name__)

col12 = styles.col12
col6 = styles.col6
hidden = styles.hidden
nostyle = styles.nostyle

layout = html.Div(style=col12,children=[
    html.Div(id=estimatorTuple[0] + "Info",children=[
        html.Div(children=[
            html.Div(style=col6,children=html.H3(children=estimatorTuple[0])),
            html.Div(style=col6,children=html.P(children=modelDescriptionDictionary[estimatorTuple[0]]))
        ]),
        html.Div(style=col6,children=[
            html.Div(id=str(estimatorTuple[0]) + "Matrix"),
            html.Div(id=str(estimatorTuple[0]) + "Metrics")
        ]),
        html.Div(style=(hidden if "knn" in estimatorTuple[0] else nostyle),children=[
            html.H4("Top 10 deciding features:"),
            dash_table.DataTable(significantFeaturesDictionary[estimatorTuple[0]])
        ]),
        html.Div(style=(nostyle if "knn" in estimatorTuple[0] else hidden),children=[
            html.H4("Top 10 deciding features:"),
            dash_table.DataTable(significantFeaturesDictionary[estimatorTuple[0]])
        ])
    ]) for estimatorTuple in allEstimatorTuples
])