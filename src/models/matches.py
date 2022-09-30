import numpy as np
import pandas as pd
import json
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import GradientBoostingClassifier as grad
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import VotingClassifier
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from dash import Dash, html, dcc
#import plotly.express as px
#import util as util

# Train the models
with open('../data/processedData/columnDataDictionary.json') as d:
    columnDataDictionary = json.load(d)
with open('../data/processedData/forestParams.json') as d:
    forestParams = json.load(d)
    preciseForestParams = forestParams["preciseForestParams"]
    recallForestParams = forestParams["recallForestParams"]  

datingTrain = pd.read_csv('../data/plotlyDashData/datingTrain.csv')
match = datingTrain["match"]
X = datingTrain.drop("match",axis=1)

sqrtn = int(np.sqrt(X.shape[0]))
logModel = lm.LogisticRegression(max_iter=1e9)
logPipe = make_pipeline(StandardScaler(), logModel)
knn5 = knn(n_neighbors=5)
knnsqrtn = knn(n_neighbors=sqrtn)
gradientdeci = grad(learning_rate=0.1)
gradientdeka = grad(learning_rate=10)
preciseForest = rf(n_estimators = preciseForestParams["n_estimators"],
                    criterion = preciseForestParams["criterion"],
                    max_depth = preciseForestParams["max_depth"],
                    max_features = preciseForestParams["max_features"])
recallForest = rf(n_estimators = recallForestParams["n_estimators"],
                  criterion = recallForestParams["criterion"],
                  max_depth = recallForestParams["max_depth"],
                  max_features = recallForestParams["max_features"])

ensembleVote = VotingClassifier(
    estimators = [
        ("logModel",logPipe),
        ("knn5",knn5),
        ("knnsqrtn",knnsqrtn),
        ("gradientdeci",gradientdeci),
        ("gradientdeka",gradientdeka),
        ("preciseForest",preciseForest),
        ("recallForest",recallForest)
    ]
)

ensembleVote.fit(X,match)