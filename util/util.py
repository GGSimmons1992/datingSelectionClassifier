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
import sys

locator = Nominatim(user_agent="datingSelectionClassifier")
geocode = RateLimiter(locator.geocode, min_delay_seconds=1.5)

locationsDictionary = dict()
if exists("../data/locations.json"):
    with open("../data/locations.json") as d:
        locationsDictionary = json.load(d)

def isNan(x):
    if type(x) != str:
        return math.isnan(x) or math.isinf(x)
    return x == 'nan'

def replaceNansWithTrainingDataValues(df):
    with open('../data/processedData/trainNanReplacementValuesDictionary.json') as d:
        trainNanReplacementValuesDictionary = json.load(d)
    for col in df.columns:
        if str(col) in trainNanReplacementValuesDictionary.keys():
            replacementValue = trainNanReplacementValuesDictionary[str(col)]
            df[col] = df[col].fillna(replacementValue)
            df[col] = df[col].replace([np.inf, -np.inf], replacementValue)
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
    with open('../data/processedData/columnDataDictionary.json') as d:
        columnDataDictionary = json.load(d)
    partnerList = columnDataDictionary['partnerList']
    
    partner_o = partnerFullDF[['iid','pid']]
    partner_o['iid_o'] = partner_o['iid']
    partner_o['pid_o'] = partner_o['pid']
    partner_o = partner_o.drop(['iid','pid'], axis=1)
    for col in list(partnerFullDF.columns):
        if col in partnerList:
            partner_o[str(col)+'_o'] = partnerFullDF[col]
    finalDF = pd.merge(candidateDF,partner_o,how='left',left_on=['iid','pid'],right_on=['pid_o','iid_o'])
    for finalCol in finalDF.columns:
        if "o_x" in finalCol:
            correctedCol = finalCol.replace("o_x","o")
            finalDF[correctedCol] = finalDF[str(finalCol)]
            finalDF = finalDF.drop(str(finalCol),axis=1)
        elif "o_y" in finalCol:
            finalDF = finalDF.drop(str(finalCol),axis=1)
    return finalDF

def returnDFWithpartnerDistance(df,datasetType,displayNoneNumber = False):
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

def getLocations(df,displayNoneNumber = False):
    milestone = datetime.datetime.now()
    nanCount = 0
    lats = []
    lons = []

    df["zipcode"] = df["zipcode"].str.replace(',', '')

    if exists("../data/coordinates.json"):
        with open("../data/coordinates.json") as d:
            coordinatesDictionary = json.load(d)
        nanCount = coordinatesDictionary["nanCount"]
        lats = coordinatesDictionary["lats"]
        lons = coordinatesDictionary["lons"]
    
    if len(lons) < df.shape[0]:
        for rowindex in range(len(lons),df.shape[0]):
            lat = np.nan
            lon = np.nan

            row = df.iloc[rowindex]
            location = getLocation(str(row['zipcode']),str(row['from']))
            if (location != None):
                lat = location[0]
                lon = location[1]

            lats.append(lat)
            lons.append(lon)

            if ((datetime.datetime.now() > milestone) and (displayNoneNumber)):
                total = df.shape[0] * 2
                nanList = [np.nan for lat in lats if isNan(lat)] + [np.nan for lon in lons if isNan(lon)]
                nanCount = len(nanList)

                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(df.shape[0])}% complete')
                print(f'{int(nanCount/2)} Non-Locations. {100 * nanCount/total}% of data is None')
                milestone += datetime.timedelta(minutes=5)
                
                coordinatesDictionary = {
                    "nanCount": nanCount,
                    "lats": lats,
                    "lons": lons
                }

                with open("../data/coordinates.json","w") as fp:
                    json.dump(coordinatesDictionary, fp)
    else:
        for rowindex in range(df.shape[0]):
            if np.isnan(lats[rowindex]) | np.isnan(lons[rowindex]):
                row = df.iloc[rowindex]
                location = getLocation(str(row['zipcode']),str(row['from']))
                if (location != None):
                    lats[rowindex] = location[0]
                    lons[rowindex] = location[1]

                if ((datetime.datetime.now() > milestone) and (displayNoneNumber)):
                    total = df.shape[0] * 2
                    nanList = [np.nan for lat in lats if isNan(lat)] + [np.nan for lon in lons if isNan(lon)]
                    nanCount = len(nanList)

                    print(datetime.datetime.now().strftime('%H:%M:%S'))
                    print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(df.shape[0])}% complete')
                    print(f'{int(nanCount/2)} Non-Locations. {100 * nanCount/total}% of data is None')
                    milestone += datetime.timedelta(minutes=5)
                    
                    coordinatesDictionary = {
                        "nanCount": nanCount,
                        "lats": lats,
                        "lons": lons
                    }

                    with open("../data/coordinates.json","w") as fp:
                        json.dump(coordinatesDictionary, fp)

    df['lats'] = pd.Series(lats)
    df['lons'] = pd.Series(lons)
    return df  

def getLocation(zipcode,fromLocation):
    locationString = zipcode
    if locationString in ['nan',None,'']:
        locationString = fromLocation
    else:
        while len(locationString)<5:
            locationString = '0' + locationString
    if locationString in locationsDictionary.keys():
        return locationsDictionary[locationString]
    try:
        coordinates = geocode(locationString).point[0:2]
        if coordinates != None:
            locationsDictionary[locationString] = coordinates
            with open("../data/locations.json","w") as fp:
                json.dump(locationsDictionary, fp)
        elif locationString != fromLocation:
            locationString = fromLocation
            coordinates = geocode(locationString).point[0:2]
            if coordinates != None:
                locationsDictionary[locationString] = coordinates
                with open("../data/locations.json","w") as fp:
                    json.dump(locationsDictionary, fp)
        return coordinates
    except BaseException:
        return None

def fixAmbiguousScores(df):
    halfwayQuestionSanityTest(df)
    df = redistributePoints(df)
    df = applyHalfwayChange(df)
    halfwayQuestionSanityTest(df)
    return df

def redistributePoints(df):
    with open('../data/processedData/columnDataDictionary.json') as d:
        columnDataDictionary = json.load(d)
    pointDistributionList = columnDataDictionary['pointDistributionList']
    
    for question in pointDistributionList:
        questionSumString = f'{question}_sum'
        questionCols = [str(col) for col in df if question in str(col)]
        df[questionSumString] = df[questionCols].sum(axis = 1)
        for questionCol in questionCols:
            questionValues = [np.nan] * df.shape[0]
            for rowindex in range(df.shape[0]):
                row = df.iloc[rowindex]
                if isNan(row[questionSumString]) | (row[questionSumString] == 0):
                    questionValues[rowindex] = row[str(questionCol)]
                else:
                    questionValues[rowindex] = row[str(questionCol)] * 100 / row[questionSumString]
            df[str(questionCol)] = pd.Series(questionValues)
        df = df.drop(questionSumString,axis=1)
    return df

def applyHalfwayChange(df):
    halfwayChangeColumns = [str(col) for col in df.columns if (("1_s" in str(col)) | ("3_s" in str(col)))]
    halfwayQuestionSanityTest(df)
    for halfwayQuestion in halfwayChangeColumns:
        targetQuestion = ""
        if ("1_s" in halfwayQuestion):
            if halfwayQuestion == "attr1_s_o":
                targetQuestion = "pf_o_att"
            elif halfwayQuestion == "sinc1_s_o":
                targetQuestion = "pf_o_sin"
            elif halfwayQuestion == "intel1_s_o":
                targetQuestion = "pf_o_int"
            elif halfwayQuestion == "fun1_s_o":
                targetQuestion = "pf_o_fun"
            elif halfwayQuestion == "amb1_s_o":
                targetQuestion = "pf_o_amb"
            elif halfwayQuestion == "shar1_s_o":
                targetQuestion = "pf_o_sha"
            else:
                targetQuestion = halfwayQuestion.replace("1_s","1_1")
        else:
            targetQuestion = halfwayQuestion.replace("3_s","3_1")
        currentMindsetAnswers = []
        for rowindex in range(df.shape[0]):
            row = df.iloc[rowindex]
            if isNan(row[halfwayQuestion]) | (row["order"] <= int(row["round"])):
                currentMindsetAnswers.append(row[targetQuestion])
            else:
                currentMindsetAnswers.append(row[halfwayQuestion])
        df[targetQuestion] = pd.Series(currentMindsetAnswers)
    df = df.drop(halfwayChangeColumns + ["round","order"],axis = 1)
    halfwayQuestionSanityTest(df)
    return df

def halfwayQuestionSanityTest(df):
    halfwayChangeColumns = [str(col) for col in df.columns if (("1_s" in str(col)) | ("3_s" in str(col)))]

    if(len(halfwayChangeColumns) > 0 and "order" not in df.columns):
        breakpoint()

def displayValueExceptionColumn(X):
    for col in X.columns:
        nansFound = np.any(np.isnan(X[col]))
        infinitesFound = np.all(np.isfinite(X[col]))
        if (nansFound or infinitesFound):
            print(col)
            print("\n")
            print(list(set(X[col])).sort())
        break