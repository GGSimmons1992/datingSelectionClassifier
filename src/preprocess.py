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
import plotly.express as px
from bdb import Breakpoint

#General code (from ensembleDash.py)
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

#matchmakers code (from matchmakers.py)
significantFeaturesDictionary = dict()
matrixDictionary = dict()
metricsDictionary = dict()

XColumns = list(X.columns)

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
    matrixDictionary[modelTuple[0]] = px.imshow(confusionMatrix,
    labels=dict(x="Actual", y="Predicted", color="Productivity"),
    x=['Match Fail', 'Match Success'],
    y=['Match Fail', 'Match Success'],
    text_auto=True)

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
        significantFeaturesDictionary[modelTuple[0]] = featureDescriptions
    elif (modelTuple[0] != "Ensemble") and ("knn" not in modelTuple[0]):
        importance = (modelTuple[1]).feature_importances_
        featureImportance = pd.DataFrame({
            "feature": XColumns,
            "featureImportance": importance
            },columns = ["feature","featureImportance"])
        featureImportancesSorted = featureImportance.sort_values(by="featureImportance", ascending=False).head(10)
        topFeatures = featureImportancesSorted["feature"].to_list()
        featureDescriptions = [descriptionDictionary[feat] for feat in topFeatures]
        significantFeaturesDictionary[modelTuple[0]] = featureDescriptions

metricsTable = dict()
for i in range(len(modelNames)):
    modelMetrics = {
        "accuracy":accuracyScores[i],
        "recall":recallScores[i],
        "precision":precisionScores[i],
    }
    metricsTable[modelNames[i]]=modelMetrics


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

#sandbox code (from sandbox.py)
halfwayQuestions = [col for col in columnList if (("1_s" in col)|("3_s" in col))]
dropList = ["iid","pid","round","order","undergra","from","zipcode","dec"] + halfwayQuestions
featureSelectValues = [col for col in columnList if col not in dropList]
featureSelectOptions = [
    {"label":descriptionDictionary[col],"value":col} for col in featureSelectValues
]
candidateDummyList = [str(k) for k in dummyDictionary.keys()]
partnerDummyList = [str(k)+"_o" for k in dummyDictionary.keys()]

candidateFeatures = []
partnerFeatures = []
for col in featureSelectValues:
    if col in partnerList:
        candidateFeatures.append(str(col))
        partnerFeatures.append(str(col)+"_o")

candidateProfile = pd.DataFrame(columns=candidateFeatures)
partnerProfile = pd.DataFrame(columns=partnerFeatures)

questionDictionary = dict()
questionDictionary["1_1"]="What do you what do you look for in a partner? (budget out of 100 points)"
questionDictionary["4_1"]="What do you what do you think others of your gender look for in a partner? (budget out of 100 points)"
questionDictionary["2_1"]="What do you what do you think others of the opposite gender look for in a partner? (budget out of 100 points)"

dummyValueDictionary = dict()
with open("../data/dummyValueDictionary.json") as d:
    dummyValueDictionary = json.load(d)

selectedValueIndex = int(np.random.uniform(0,len(featureSelectOptions)))
while selectedValueIndex == len(featureSelectOptions):
    selectedValueIndex = int(np.random.uniform(0,len(featureSelectOptions)))

selectedValue = featureSelectOptions[selectedValueIndex]

selectedMatchIndex = int(np.random.uniform(0,datingTest.shape[0]))
while selectedMatchIndex == datingTest.shape[0]:
    selectedMatchIndex = int(np.random.uniform(0,datingTest.shape[0]))

selectedMatchDF = datingTest.iloc[[selectedMatchIndex]]
selectedMatch = datingTest.iloc[selectedMatchIndex]
for col in selectedMatchDF.columns:
    if col in candidateProfile.columns:
        if col in candidateDummyList:
            dummyCols = dummyDictionary[col]
            for dummyCol in dummyCols:
                if selectedMatch[dummyCol] == 1:
                    candidateProfile[col] = dummyValueDictionary[dummyCol]
                if selectedMatch[str(dummyCol)+"_o"] == 1:
                    partnerProfile[str(dummyCol)+"_o"] = dummyValueDictionary[str(dummyCol)+"_o"]
        elif col=="gender":
            candidateProfile[col] = "female" if selectedMatch[col] == 0 else "male"
            partnerProfile[col+"_o"] = "female" if selectedMatch[col+"_o"] == 0 else "male"
        else:
            candidateProfile[col] = selectedMatch[col]
            partnerProfile[str(col)+"_o"] = selectedMatch[str(col)+"_o"]