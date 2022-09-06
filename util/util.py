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
    milestone = datetime.datetime.now()
    nanCount=0
    distances = []
    if exists(f"../data/{datasetType}Distances.json"):
        with open(f"../data/{datasetType}Distances.json") as d:
            distancesDictionary = json.load(d)
        nanCount = distancesDictionary["nanCount"]
        distances = distancesDictionary["distances"]

    if len(distances) < df.shape[0]:
        for rowindex in range(len(distances),df.shape[0]):
            row = df.iloc[rowindex]
            candidateLocation = getLocation(str(row['zipcode']),str(row['from']))
            partnerLocation = getLocation(str(row['zipcode_o']),str(row['from_o']))
            if(candidateLocation != None and partnerLocation != None):
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

                with open(f"../data/{datasetType}Distances.json", 'w') as fp:
                    json.dump(distancesDictionary, fp)
    else:
        if displayNoneNumber:
            print('starting doublecheck')
        for rowindex in range(df.shape[0]):
            if(distances[rowindex] == None):
                row = df.iloc[rowindex]
                candidateLocation = getLocation(str(row['zipcode']),str(row['from']))
                partnerLocation = getLocation(str(row['zipcode_o']),str(row['from_o']))
                if(candidateLocation != None and partnerLocation != None):
                    distance = great_circle(candidateLocation,partnerLocation).mi
                    if distance != None:
                        distances[rowindex] = distance
                        nanCount -= 1

                if ((datetime.datetime.now() > milestone) and (displayNoneNumber)):
                    total = df.shape[0]
                    print(datetime.datetime.now().strftime('%H:%M:%S'))
                    print(f'{100 * nanCount/total}% of data is None')
                    milestone += datetime.timedelta(minutes=5)
                    
                    distancesDictionary = {
                        "nanCount": nanCount,
                        "distances": distances
                    }

                    with open(f"../data/{datasetType}Distances.json", 'w') as fp:
                        json.dump(distancesDictionary, fp)
    
    df['partnerDistance'] = pd.Series(distances)
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
