import pandas as pd
import json
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import styles

with open("../data/plotlyDashData/sandboxCollection.json") as d:
    sandboxCollection = json.load(d)
    featureSelectOptions = sandboxCollection["featureSelectOptions"]
    selectedValue = sandboxCollection["selectedValue"]
    descriptionDictionary = sandboxCollection["descriptionDictionary"]
    selectedMatchFeatures = sandboxCollection["selectedMatchFeatures"]
    selectedMatch = pd.Series(sandboxCollection["selectedMatch"],index=selectedMatchFeatures)
    candidateFeatures = sandboxCollection["candidateFeatures"]
    partnerFeatures = sandboxCollection["partnerFeatures"]
    questionDictionary  = sandboxCollection["questionDictionary"]

candidateProfile = pd.read_csv("../data/plotlyDashData/candidateProfile.csv")
partnerProfile = pd.read_csv("../data/plotlyDashData/partnerProfile.csv")

dash.register_page(__name__,path="/")

col12 = styles.col12
col8 = styles.col8
col6 = styles.col6
col4 = styles.col4
col3 = styles.col3
fitContent = styles.fitContent
displayBlock = styles.displayBlock

featureSelect = dcc.Dropdown(
    id="featureSelect",
    options=featureSelectOptions,
    value=[]
)

featureNumber = html.Div(id='featureNumber',style=displayBlock,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Input(id="featureNumberInput",type="number")
])

featureDropdown = html.Div(id='featureNumber',style=displayBlock,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Dropdown(id='featureDropdownInput')
])

featureQuestion = html.Div(id='featureQuestion',style=displayBlock,children=[
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
        html.Span(id="questionTotal",children="100")
    ]),
    html.Button(id="submitQuestion",children="submit")
])

questionModal = dbc.Modal(id='modal',children=[
    featureQuestion
])

layout = html.Div(style=col12,children=[
    html.Div(children=[
        html.Div(style=col4,children=[
            html.H3(style=displayBlock,children="What do you want to examine"),
            featureSelect
        ]),
        html.Div(style=col8,children=[
            html.H3(style=col12,children="Calculated Values: Edit profile traits to change these"),
            html.Div(style=col6,children=[
                html.Div(style=col12,title=descriptionDictionary["partnerDistance"],children=[
                    html.Span(children="Partner Distance: "),
                    html.Span(id="partnerDistance",children=str(round(selectedMatch["partnerDistance"]))),
                    html.Span(children="miles")
                ]),
                html.Div(style=col12,title=descriptionDictionary["samerace"],children=[
                    html.Span(children="Same race?: "),
                    html.Span(id="samerace",children = "yes" if (selectedMatch["samerace"]==1) else "no")
                ])
            ]),
            html.Div(style=col6,children=[
                html.Div(style=col6,id="sharedInterestsValue",children=[
                    html.Span(children="Shared Interest Correlation:"),
                    html.Span(id="int_corr",children=selectedMatch["int_corr"])
                ]),
                html.Div(style=col6,children=dcc.Graph(style=col12,id="sharedInterestsGraph"))
            ])
        ])
        
    ]),
    html.Div(children=[
        html.Div(style=col3,children=[
            html.H3(style=displayBlock,children="candidate features"),
            html.Div(children=[
                html.Div(id=str(col)+"Display",title=descriptionDictionary[col],children = [
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
            html.H3(style=displayBlock,children="partner features"),
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

