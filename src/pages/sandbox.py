import numpy as np
import pandas as pd
import json
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import styles
import ensembleDash

datingTest = ensembleDash.datingTest
descriptionDictionary = ensembleDash.descriptionDictionary
dummyDictionary = ensembleDash.dummyDictionary
columnList = ensembleDash.columnList
partnerList = ensembleDash.partnerList
originalEstimtatorTuples = ensembleDash.originalEstimtatorTuples

#featureSelectOptions = ensembleDash.featureSelctOptions
#candidateFeatures = ensembleDash.candidateFeatures
#partnerFeatures = ensembleDash.partnerFeatures

dropList = ["iid","pid","round","order","undergra","from","zipcode","dec"]
featureSelectValues = [col for col in columnList if col not in dropList]
featureSelectOptions = [
    {"label":descriptionDictionary[col],"value":col} for col in featureSelectValues
]
candidateDummyList = [str(k) for k in dummyDictionary.keys()]
partnerDummyList = [str(k)+"_o" for k in dummyDictionary.keys()]

candidateFeatures = []
partnerFeatures = []
for col in columnList:
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
for k in dummyDictionary.keys():
    dummyCategoryDictionary = dict()
    for dummyCol in dummyDictionary[k]:
        print(dummyCol)
        dummyValue = Input()
        dummyValueDictionary[dummyCol]=dummyValue

selectedValueIndex = int(np.random.uniform(0,len(featureSelectOptions)))
while selectedValueIndex == len(featureSelectOptions):
    selectedValueIndex = int(np.random.uniform(0,len(featureSelectOptions)))

selectedValue = featureSelectOptions[selectedValueIndex]

selectedMatchIndex = int(np.random.uniform(0,datingTest.shape[0]))
while selectedMatchIndex == datingTest.shape[0]:
    selectedMatchIndex = int(np.random.uniform(0,datingTest.shape[0]))

selectedMatch = datingTest.iloc[selectedMatchIndex]

for col in candidateFeatures:
    if col in candidateDummyList:
        dummyCols = dummyDictionary[col]
        for dummyCol in dummyCols:
            if selectedMatch[dummyCol] == 1:
                candidateProfile[col] = dummyValueDictionary[dummyCol]
            if selectedMatch[str(dummyCol)+"_o"] == 1:
                partnerProfile[str(dummyCol)+"_o"] = dummyValueDictionary[str(dummyCol)+"_o"]
    else:
        candidateProfile[col] = selectedMatch[col]
        partnerProfile[str(col)+"_o"] = selectedMatch[str(col)+"_o"]

Dash.register_page(__name__,path="/")

hidden = styles.hidden
col12 = styles.col12
col6 = styles.col6
col4 = styles.col4
col3 = styles.col3

featureSelct = dcc.Dropdown(
    id="featureSelect",
    options=featureSelectOptions,
    value=selectedValue["value"]
)

featureNumber = html.Div(id='featureNumber',style=hidden,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Input(id="featureNumberInput",type="number")
])

featureDropdown = dcc.Dropdown(id='featureDropdown',style=hidden)

featureQuestion = html.Div(id='featureQuestion',style=hidden,children=[
    html.Div(id="question"),
    html.Div("attractiveness"),
    dcc.Input(id="attr",type="number"),
    html.Div("sincerity"),
    dcc.Input(id="sinc",type="number"),
    html.Div("intelligence"),
    dcc.Input(id="int",type="number"),
    html.Div("fun"),
    dcc.Input(id="fun",type="number"),
    html.Div("shared interests"),
    dcc.Input(id="shar",type="number"),
    html.Br,
    html.Div(children=[
        "total = ",
        html.span(id="questionTotal",children="100")
    ]),
    html.Button(id="submitQuestion",children="submit")
])

modal = dcc.Modal(id='modal',children=[
    featureNumber,
    featureDropdown,
    featureQuestion
])

layout = html.Div(style=col12,children=[
    html.Div(children=[
        html.Div(style=col4,children=[
            featureSelct
        ]),
        html.Div(style=col4,children=[
            html.Div(title=descriptionDictionary["partnerDistance"],children=[
                html.Span(children="Partner Distance: "),
                html.Span(id="partnerDistance",children=str(round(selectedMatch["partnerDistance"]))),
                html.Span(children="miles")
            ]),
            html.Div(title=descriptionDictionary["samerace"],children=[
                html.Span(children="Same race?: "),
                html.Span(id="samerace",children = 
                "yes" if (selectedMatch["samerace"]==1) else "no")
            ])
        ]),
        html.Div(style=col4,children=[
            html.Div(id="sharedInterestsValue",children=[
                html.Span(children="Shared Interest Correlation:"),
                html.Span(id="int_corr",children=selectedMatch["int_corr"])
            ]),
            html.Div(children=dcc.Graph(id="sharedInterestsGraph"))
        ])
    ]),
    html.Div(children=[
        html.Div(style=col3,children=[
            html.H3("candidate features"),
            html.Div(children=[
                html.Div(id=str(col)+"Edit",title=descriptionDictionary[col],children = [
                    html.Span(children=f"{col}: "),
                    html.Span(id=str(col)+"Value",children=candidateProfile[col])
                ]) 
                for col in candidateFeatures
            ])
        ]),
        html.Div(style=col6,children=[
            html.Div(children=dcc.Graph(style=col12,id="predictProbaGraph")),
            html.Div(children=dcc.Graph(style=col12,id="diversityGraph"))
        ]),
        html.Div(style=col3,children=[
            html.H3("partner features"),
            html.Div(children=[
                html.Div(id=str(col)+"Edit",title=descriptionDictionary[col],children = [
                    html.Span(children=f"{col}: "),
                    html.Span(id=str(col)+"Value",children=partnerProfile[col])
                ]) 
                for col in partnerFeatures
            ])
        ])
    ])
])

