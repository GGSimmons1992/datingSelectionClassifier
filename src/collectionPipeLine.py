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

#General code (from ensembleDash.py)
# Train the models
with open('../data/plotlyDashData/columnDataDictionary.json') as d:
    columnDataDictionary = json.load(d)
    columnList = columnDataDictionary['columnList']
    nonBinaryCategoricalList = columnDataDictionary['nonBinaryCategoricalList']
    stringToFloatList = columnDataDictionary['stringToFloatList']
    pointDistributionList = columnDataDictionary['pointDistributionList']
    sharedList = columnDataDictionary['sharedList']
    partnerList = columnDataDictionary['partnerList']
with open("../data/plotlyDashData/dummyDictionary.json") as d:
    dummyDictionary = json.load(d)
with open('../data/plotlyDashData/treeParams.json') as d:
    treeParams = json.load(d)
    preciseTreeParams = treeParams["preciseTreeParams"]
    recallTreeParams = treeParams["recallTreeParams"]
with open('../data/plotlyDashData/forestParams.json') as d:
    forestParams = json.load(d)
    preciseForestParams = forestParams["preciseForestParams"]
    recallForestParams = forestParams["recallForestParams"]
with open("../descriptionDictionary.json") as d:
    descriptionDictionary = json.load(d)  

datingTrain = pd.read_csv('../data/plotlyDashData/datingTrain.csv')
datingTest = pd.read_csv('../data/plotlyDashData/datingTest.csv')
datingFull = pd.read_csv('../data/plotlyDashData/datingFull.csv')

match = datingTrain["match"]
X = datingTrain.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])
matchTest = datingTest["match"]
XTest = datingTest.drop("match",axis=1).select_dtypes(include=['uint8','int64','float64'])

datingMale = datingFull[datingFull["gender"]==1].drop("match",axis=1)
datingFemale = datingFull[datingFull["gender"]==0].drop("match",axis=1)
datingFull = datingFull.drop("match",axis=1)

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

ensembleVote = VotingClassifier(estimators = originalEstimtatorTuples,voting="soft")
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
    matrixDictionary[modelTuple[0]] = confusionMatrix.tolist()

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
    "knnsqrtn":"k-neighbors with k=sqrt(n_features)",
    "gradientdeci":"gradient boosting classifer with learning rate = 0.1",
    "gradientdeka":"gradient boosting classifer with learning rate = 10",
    "preciseTree":"decision tree trained for best precision" + str(preciseTreeParams),
    "recallTree":"decision tree trained for best recall" + str(recallTreeParams),
    "preciseForest":"random forest trained for best precision" + str(preciseForestParams),
    "recallForest":"random forest trained for best recall" + str(recallForestParams)
}

#sandbox code (from sandbox.py)
halfwayQuestions = [col for col in columnList if (("1_s" in col)|("3_s" in col))]
dropList = ["iid","pid","round","order","undergra","from","zipcode","dec","match","gender"] + halfwayQuestions
featureSelectValues = [col for col in columnList if col not in dropList] + ["partnerDistance"]
featureSelectValues += [str(col)+"_o" for col in featureSelectValues if col not in ["samerace","int_corr","partnerDistance"]]
featureSelectOptions = [
    {"label":f"{col}) {descriptionDictionary[col]}","value":col} for col in featureSelectValues
]

# store into json collection

collectionDictionary = dict()

collectionDictionary["modelDescriptionDictionary"] = modelDescriptionDictionary
collectionDictionary["matrixDictionary"] = matrixDictionary
collectionDictionary["metricsTable"] = metricsTable
collectionDictionary["significantFeaturesDictionary"] = significantFeaturesDictionary
collectionDictionary["featureSelectOptions"] = featureSelectOptions

with open("../data/plotlyDashData/collectionDictionary.json","w") as fp:
    json.dump(collectionDictionary,fp)

for genderTuple in [("male",datingMale),("female",datingFemale),("overall",datingFull)]:
    predictProbaDictionary = dict()
    for mod in allEstimatorTuples:
        predictProbaDictionary[mod[0]] = np.array(mod[1].predict_proba(genderTuple[1])[:,1]).reshape(-1,).tolist()
    pd.DataFrame(predictProbaDictionary,
    columns=list(predictProbaDictionary.keys())).to_csv(
        f"../data/plotlyDashData/{genderTuple[0]}Predictions.csv")