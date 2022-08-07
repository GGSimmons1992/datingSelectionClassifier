import numpy as np
import pandas as pd
import json
import geopy
from geopy import GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
from os.path import exists

with open('../data/googleApiKey.json') as d:
    googleKeyDictionary = json.load(d)
    apiKey = googleKeyDictionary['apikey']
locator = GoogleV3(api_key=apiKey)
geocode = RateLimiter(locator.geocode, min_delay_seconds=1.5)

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
