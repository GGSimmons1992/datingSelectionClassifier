from bdb import Breakpoint
import numpy as np
import pandas as pd
import json
import geopy
import math
import sklearn.linear_model as lm
import sklearn.metrics as sm
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

def hasInfOrNanValues(arr):
    np.isnan(arr).any()

def switchNumbersAndCategoriesFromRawData(df):
    df = castStringNumberAsFloat(df)
    df = stringifyCategoricalColumns(df)
    return df

def castStringNumberAsFloat(df):
    with open('../data/processedData/columnDataDictionary.json') as d:
        columnDataDictionary = json.load(d)
    dfColumns = df.columns
    for col in dfColumns:
        if str(col) in columnDataDictionary["stringToFloatList"]:
            df[str(col)] = df[str(col)].str.replace(',', '').astype(float)
    return df

def stringifyCategoricalColumns(df):
    
    with open('../data/processedData/columnDataDictionary.json') as d:
        columnDataDictionary = json.load(d)
    dfColumns = list(df.columns)
    for col in dfColumns:
        stringcol = str(col)
        if stringcol in columnDataDictionary["nonBinaryCategoricalList"]:
            df[stringcol] = df[stringcol].astype(str)
    return df

def replaceNansWithTrainingDataValues(df):
    with open('../data/processedData/trainNanReplacementValuesDictionary.json') as d:
        trainNanReplacementValuesDictionary = json.load(d)
    numericColumns = df.select_dtypes(include=['uint8','int64','float64']).columns
    for col in numericColumns:
        replacementValue = 0
        if str(col) in trainNanReplacementValuesDictionary.keys():
            replacementValue = trainNanReplacementValuesDictionary[str(col)]
            df[col] = df[col].fillna(replacementValue)
            df[col] = df[col].replace([np.inf, -np.inf], replacementValue)
            nanMask = np.isnan(df[col]) | (np.isfinite(df[col]) == False)
            df.loc[nanMask,col] = replacementValue
    displayValueExceptionColumn(df)
    return df

def addDummies(df):
    dummyDictionary = dict()
    if exists("../data/processedData/dummyDictionary.json"):
        with open("../data/processedData/dummyDictionary.json") as d:
            dummyDictionary = json.load(d)
    
    categoricalData = df.select_dtypes(include=['O'])
    for col in categoricalData.columns:
        df[col]=df[col].fillna('nan')
    
    for col in categoricalData.columns:
        dummyData = pd.get_dummies(df[col],prefix=col,drop_first=False)
        if len(dummyData.columns) <= 25:
            if col == "race": #no native americans, thus excluded from original dummification method
                dummyData["race_5.0"] = 0 
            df = pd.concat([df,dummyData],axis=1)
            if str(col) not in dummyDictionary.keys():
                dummyDictionary[str(col)] = list(dummyData.columns)
    
    with open("../data/processedData/dummyDictionary.json","w") as fp:
        json.dump(dummyDictionary,fp)

    return df

def plotCorrelation(x,y,title):
    model = lm.LinearRegression()
    model.fit(x,y)
    accScore = model.score(x,y)
    m = model.coef_[0]
    b = model.intercept_
    fig = plt.figure()
    plotX = np.linspace(np.min(x),np.max(x))
    plt.scatter(x,y)
    plt.plot(plotX,m*plotX+b)
    plt.title(f'{title} m={np.round(m,2)}, b={np.round(b,2)},accuracy={accScore}')
    
def joinToPartner(candidateDF,partnerFullDF):
    # lines with * comment were added to resolve "PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance"
    # lines with ** comment were editted to accommodate comment above
    # analyze commit closest to 18:20 8 October 2022 for more details
    with open('../data/processedData/columnDataDictionary.json') as d:
        columnDataDictionary = json.load(d)
    partnerList = columnDataDictionary['partnerList']
    nonBinaryCategoricalList = columnDataDictionary['nonBinaryCategoricalList']
    
    partner_o = partnerFullDF[['iid','pid']]
    partner_o['iid_o'] = partner_o['iid']
    partner_o['pid_o'] = partner_o['pid']
    partner_o = partner_o.drop(['iid','pid'], axis=1)
    partner_oCols = partner_o.columns # *
    partner_oDictionary = dict() # *
    for col in partner_oCols: # *
        partner_oDictionary[col] = list(partner_o[col]) # *
    partner_o = partner_oDictionary # *

    for col in list(partnerFullDF.columns):
        if col in partnerList:
            partner_o[str(col)+'_o'] = list(partnerFullDF[col]) # **
            if col in nonBinaryCategoricalList:
                categoricalValues = partner_o[str(col)+'_o'] #*
                partner_o[str(col)+'_o'] = [str(val) for val in categoricalValues] #**
                if ((str(col)+'_o') not in nonBinaryCategoricalList and '_o_o' not in (str(col)+'_o')):
                    nonBinaryCategoricalList.append(str(col)+'_o')
    
    partner_o = pd.DataFrame(partner_o,columns=list(partner_o.keys())) # *

    columnDataDictionary['nonBinaryCategoricalList'] = nonBinaryCategoricalList

    with open('../data/processedData/columnDataDictionary.json', 'w') as fp:
        json.dump(columnDataDictionary, fp)
    
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
    firstIteration = True
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
                if firstIteration == False:
                    print(datetime.datetime.now().strftime('%H:%M:%S'))
                    print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(total)}% complete')
                    print(f'{100 * nanCount/total}% of data is None')
                milestone += datetime.timedelta(minutes=5)
                firstIteration = False
                
                distancesDictionary = {
                    "nanCount": nanCount,
                    "distances": distances
                }

                with open(f"../data/processedData/{datasetType}Distances.json", 'w') as fp:
                    json.dump(distancesDictionary, fp)
    
    df['partnerDistance'] = pd.Series(distances)
    
    if displayNoneNumber:
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(total)}% complete')
        print(f'{100 * nanCount/total}% of data is None')

    return df

def getLocations(df,displayNoneNumber = False):
    milestone = datetime.datetime.now()
    firstIteration = True
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
                if(firstIteration == False):
                    print(datetime.datetime.now().strftime('%H:%M:%S'))
                    print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(df.shape[0])}% complete')
                    print(f'{int(nanCount/2)} Non-Locations. {100 * nanCount/total}% of data is None')
                milestone += datetime.timedelta(minutes=5)
                firstIteration = False
                
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

                    if(firstIteration == False):
                        print(datetime.datetime.now().strftime('%H:%M:%S'))
                        print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(df.shape[0])}% complete')
                        print(f'{int(nanCount/2)} Non-Locations. {100 * nanCount/total}% of data is None')
                    milestone += datetime.timedelta(minutes=5)
                    firstIteration = False
                    
                    coordinatesDictionary = {
                        "nanCount": nanCount,
                        "lats": lats,
                        "lons": lons
                    }

                    with open("../data/coordinates.json","w") as fp:
                        json.dump(coordinatesDictionary, fp)

    df['lats'] = pd.Series(lats)
    df['lons'] = pd.Series(lons)

    if displayNoneNumber:
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        print(f'{rowindex} of {df.shape[0]}: {rowindex*100.0/(df.shape[0])}% complete')
        print(f'{int(nanCount/2)} Non-Locations. {100 * nanCount/total}% of data is None')

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
    halfwayQuestionSanityTest(df,"beginning of fixAmbiguousScores")
    df = redistributePoints(df)
    df = applyHalfwayChange(df)
    halfwayQuestionSanityTest(df,"end of fixAmbiguousScores")
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
                if np.isnan(row[questionSumString]) | (row[questionSumString] == 0):
                    questionValues[rowindex] = row[str(questionCol)]
                else:
                    questionValues[rowindex] = row[str(questionCol)] * 100.0 / row[questionSumString]
                    if ((np.isnan(questionValues[rowindex]) == False) & np.isfinite(questionValues[rowindex])):
                        questionValues[rowindex] = round(questionValues[rowindex])
            df[str(questionCol)] = pd.Series(questionValues)
        df = df.drop(questionSumString,axis=1)
    return df

def applyHalfwayChange(df):
    halfwayChangeColumns = [str(col) for col in df.columns if (("1_s" in str(col)) | ("3_s" in str(col)))]
    halfwayQuestionSanityTest(df,"beginning of applyHalfwayChange")
    for halfwayQuestion in halfwayChangeColumns:
        targetQuestion = ""
        if ("1_s" in halfwayQuestion):
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
    dropColumns = halfwayChangeColumns + ["round","order"]
    for blindDatePartnerId in ["round_o","order_o"]:
        if blindDatePartnerId in df.columns:
            dropColumns.append(blindDatePartnerId)
    df = df.drop(dropColumns,axis = 1)
    halfwayQuestionSanityTest(df,"end of applyHalfwayChange")
    return df

def halfwayQuestionSanityTest(df,location):
    halfwayChangeColumns = [str(col) for col in df.columns if (("1_s" in str(col)) | ("3_s" in str(col)))]
    if(len(halfwayChangeColumns) > 0 and "order" not in df.columns):
        print(location)
        breakpoint()

def displayValueExceptionColumn(X):
    turnOnBreakpoint = False
    for col in X.select_dtypes(include=['uint8','int64','float64']).columns:
        nansFound = np.any(np.isnan(X[col]))
        infinitesFound = np.all(np.isfinite(X[col])) == False
        if (nansFound or infinitesFound):
            turnOnBreakpoint = True
            print(col)
            print("\n")
            sortedList = sorted(list(set(X[col])))
            print(sortedList)
    if turnOnBreakpoint:
        breakpoint()

def displayMetricScores(yTest,yPredict,modelName):
    print(f"{modelName} accuracy score: {sm.accuracy_score(yTest,yPredict)}")
    print(f"{modelName} recall score: {sm.recall_score(yTest,yPredict)}")
    print(f"{modelName} precision score: {sm.precision_score(yTest,yPredict)}")

def displayFeatureImportances(columns,fittedModel,modelName):
    importance = fittedModel.feature_importances_
    featureImportance = pd.DataFrame({
        "feature": columns,
        "featureImportance": importance
        },columns = ["feature","featureImportance"])
    featureImportancesSorted = featureImportance.sort_values(by="featureImportance", ascending=False)
    print(f'{modelName} top 10 feature importances')
    for i in range(10):
        featureRow = featureImportancesSorted.iloc[i]
        feature = featureRow['feature']
        featureValue = featureRow['featureImportance']
        print(f'Rank {i}: {feature}: score: {featureValue}')
    print("\n")

def estimateIncomeByLocation(zipcode,fromLocation):
    # DO NOT USE FOR NAN REPLACEMENT! THIS IS FOR ESTIMATING PLOTLY DASH USER INCOME BY LOCATION!!!
    incomeByLocationDictionary = dict()
    locationDictionary = dict()
    defaultIncome = np.nan
    with open("../data/incomeByLocationDictionary.json") as d:
        incomeByLocationDictionary = json.load(d)
    with open("../data/locations.json") as d:
        locationDictionary = json.load(d)
    with open("../data/processedData/trainNanReplacementValuesDictionary.json") as d:
        defaultIncome = (json.load(d))["income"]
    
    candidateLocation = getLocation(zipcode,fromLocation)
    if candidateLocation == None:
        return defaultIncome

    knownIncomes = []
    distances = []
    knownLocationsWithIncomes = incomeByLocationDictionary.keys()
    for knownLocation in knownLocationsWithIncomes:
        knownLocationCoordinates = (locationDictionary[0],locationDictionary[1])
        distance = great_circle(candidateLocation,knownLocationCoordinates).mi
        if distance != None:
            knownIncomes.append(incomeByLocationDictionary[knownLocation])
            distances.append(distance)
    
    distancesAndIncomes = pd.DataFrame({
        "knownIncomes": knownIncomes,
        "distances":distances
    },columns=["knownIncomes","distances"])
    closest3Incomes = distancesAndIncomes.sort_values(by="distances",ascending=True).head(3)
    return closest3Incomes[knownIncomes].mean()