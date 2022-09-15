from bdb import Breakpoint
import numpy as np
import pandas as pd
import json
import geopy
import math
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
from os.path import exists
import datetime

locator = Nominatim(user_agent="datingSelectionClassifier")
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
    partner_o['iid_o'] = partner_o['iid']
    partner_o['pid_o'] = partner_o['pid']
    partner_o = partner_o.drop(['iid','pid'], axis=1)
    for col in list(partnerFullDF.columns):
        if col in partnerList:
            partner_o[str(col)+'_o'] = partnerFullDF[col]

    return pd.merge(candidateDF,partner_o,how='left',left_on=['iid','pid'],right_on=['pid_o','iid_o'])

def returnDFWithpartnerDistance(df,datasetType,displayNoneNumber = False):
    df = getCoupleLocations(df,datasetType,displayNoneNumber)
    
    milestone = datetime.datetime.now()
    nanCount=0
    distances = []
    if exists(f"../data/processedData/{datasetType}Distances.json"):
        with open(f"../data/processedData/{datasetType}Distances.json") as d:
            distancesDictionary = json.load(d)
        nanCount = distancesDictionary["nanCount"]
        distances = distancesDictionary["distances"]

    if len(distances) < df.shape[0]:
        for rowindex in range(len(distances),df.shape[0]):
            row = df.iloc[rowindex]
            if (isNan(row["lats"]) == False and isNan(row["lons"]) == False and 
            isNan(row["lats_o"]) == False and isNan(row["lons_o"]) == False):
                candidateLocation = (row["lats"],row["lons"])
                partnerLocation = (row["lats_o"],row["lons_o"])
                distance = great_circle(candidateLocation,partnerLocation).mi
                if distance == None:
                    nanCount += 1
                distances.append(distance)
            else:
                distances.append(np.nan)
                nanCount += 1
            if ((datetime.datetime.now() > milestone) and (displayNoneNumber)):
                total = df.shape[0]
                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(total)}% complete')
                print(f'{100 * nanCount/total}% of data is None')
                milestone += datetime.timedelta(minutes=5)
                
                distancesDictionary = {
                    "nanCount": nanCount,
                    "distances": distances
                }

                with open(f"../data/processedData/{datasetType}Distances.json", 'w') as fp:
                    json.dump(distancesDictionary, fp)
    
    df['partnerDistance'] = pd.Series(distances)
    return df

def getCoupleLocations(df,datasetType,displayNoneNumber = False):
    milestone = datetime.datetime.now()
    nanCount = 0
    candidateLats = []
    candidateLons = []
    partnerLats = []
    partnerLons = []
    if exists(f"../data/processedData/{datasetType}Locations.json"):
        with open(f"../data/processedData/{datasetType}Locations.json") as d:
            locationsDictionary = json.load(d)
        nanCount = locationsDictionary["nanCount"]
        candidateLats = locationsDictionary["candidateLats"]
        candidateLons = locationsDictionary["candidateLons"]
        partnerLats = locationsDictionary["partnerLats"]
        partnerLons = locationsDictionary["partnerLons"]
    
    if len(partnerLons) < df.shape[0]:
        for rowindex in range(len(partnerLons),df.shape[0]):
            candidateLat = np.nan
            candidateLon = np.nan
            partnerLat = np.nan
            partnerLon = np.nan

            row = df.iloc[rowindex]
            candidateLocation = getLocation(str(row['zipcode']),str(row['from']))
            partnerLocation = getLocation(str(row['zipcode_o']),str(row['from_o']))
            if (candidateLocation != None):
                candidateLat = candidateLocation[0]
                candidateLon = candidateLocation[1]
            if (partnerLocation != None):
                partnerLat = partnerLocation[0]
                partnerLon = partnerLocation[1]
            
            candidateLats.append(candidateLat)
            candidateLons.append(candidateLon)
            partnerLats.append(partnerLat)
            partnerLons.append(partnerLon)

            if ((datetime.datetime.now() > milestone) and (displayNoneNumber)):
                total = df.shape[0] * 4
                nanCount = candidateLats.count(np.nan) + candidateLons.count(np.nan) + partnerLats.count(np.nan) + partnerLons.count(np.nan)

                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(df.shape[0])}% complete')
                print(f'{100 * nanCount/total}% of data is None')
                milestone += datetime.timedelta(minutes=5)
                
                locationsDictionary = {
                    "nanCount": nanCount,
                    "candidateLats": candidateLats,
                    "candidateLons": candidateLons,
                    "partnerLats": partnerLats,
                    "partnerLons": partnerLons
                }

                with open(f"../data/processedData/{datasetType}Locations.json", 'w') as fp:
                    json.dump(locationsDictionary, fp)
    
    df['lats'] = pd.Series(candidateLats)
    df['lons'] = pd.Series(candidateLons)
    df['lats_o'] = pd.Series(partnerLats)
    df['lons_o'] = pd.Series(partnerLons)
    return df  

def getLocation(zipcode,fromLocation):
    try:
        if zipcode in ['nan',None,'']:
            if type(fromLocation) == str:
                return geocode(fromLocation).point[0:2]
            return None
        while len(zipcode)<5:
            zipcode='0'+zipcode
        return geocode(zipcode).point[0:2]
    except BaseException:
        return None
