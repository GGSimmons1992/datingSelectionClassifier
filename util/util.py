import numpy as np
import pandas as pd
import json
import geopy
import math
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from geopy import GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
from os.path import exists

with open('../data/googleApiKey.json') as d:
    googleKeyDictionary = json.load(d)
    apiKey = googleKeyDictionary['apikey']
locator = GoogleV3(api_key=apiKey)
geocode = RateLimiter(locator.geocode, min_delay_seconds=1.5)

def isNan(x):
    if type(x) != str:
        return math.isnan(x) or math.isinf(x)
    return x == 'nan'

def replaceNansWithTrainingDataValues(df):
    with open('../data/trainNanReplacementValuesDictionary.json') as d:
        trainNanReplacementValuesDictionary = json.load(d)
    for col in df.columns:
        df[col] = df[col].fillna(trainNanReplacementValuesDictionary[str(col)])
    return df

def removeDummiesAndCorrelatedFeaturesFromAvailabilityList(availabilityList,feature):
    with open('../data/sigCorrDictionary.json') as d:
        sigCorrDictionary = json.load(d)
    with open('../data/relatedDummiesDictionary.json') as d:
        relatedDummiesDictionary = json.load(d)
    if (feature in relatedDummiesDictionary.keys()):
        for dummy in relatedDummiesDictionary[feature]:
            if dummy in availabilityList:
                availabilityList.remove(dummy)
    for corrFeature in sigCorrDictionary[feature]:
        if corrFeature in availabilityList:
            availabilityList.remove(corrFeature)
    if feature in availabilityList:
        availabilityList.remove(feature)
    return availabilityList

def hasInfOrNanValues(arr):
    np.isnan(arr).any()

def plotCorrelation(x,y,title):
    model = lm.LinearRegression()
    model.fit(x,y)
    m = model.coef_[0]
    b = model.intercept_
    fig = plt.figure()
    plotX = np.linspace(np.min(x),np.max(x))
    plt.scatter(x,y)
    plt.plot(plotX,m*plotX+b)
    plt.title(f'{title} m={np.round(m,2)}, b={np.round(b,2)}')
    
def joinToPartner(candidateDF,partnerFullDF):
    with open('../data/columnDataDictionary.json') as d:
        columnDataDictionary = json.load(d)
    partnerList = columnDataDictionary['partnerList']
    
    partner_o = partnerFullDF[['iid','pid']]
    for col in list(partner_o.columns):
        if col in partnerList:
            partner_o[str(col)+'_o'] = partnerFullDF[col]
    
    return pd.merge(candidateDF,partner_o,how='left',left_on=['iid','pid'],right_on=['pid','iid'])


