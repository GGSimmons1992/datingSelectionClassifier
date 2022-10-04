import numpy as np
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import styles
import ensembleDash

datingTest = ensembleDash.datingTest
featureSelectOptions = ensembleDash.featureSelctOptions
candidateFeatures = ensembleDash.candidateFeatures
partnerFeatures = ensembleDash.partnerFeatures

selectedValueIndex = int(np.random.uniform(0,len(featureSelectOptions)))
while selectedValueIndex == len(featureSelectOptions):
    selectedValueIndex = int(np.random.uniform(0,len(featureSelectOptions)))

selectedValue = featureSelectOptions[selectedValueIndex]

Dash.register_page(__name__,path="/")

hidden = styles.hidden
col12 = styles.col12
col6 = styles.col6
col4 = styles.col4
col3 = styles.col3

featureSelct = dcc.Dropdown(
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
            html.Div(id="samerace"),
            html.Div(id="partnerDistance")
        ]),
        html.Div(style=col4,children=[
            html.Div(id="sharedInterestsValue",children=[
                html.Span(children="Shared Interest Correlation:"),
                html.Span(id="int_corr")
            ]),
            html.Div(children=dcc.Graph(id="sharedInterestsGraph"))
        ])
    ]),
    html.Div(children=[
        html.Div(style=col3,children=[
            html.H3("candidate features"),
            html.Div(children=[
                html.Div(
                    html.Span(children=f"{col}: "),
                    html.Span(candidateProfile[col])
                ) 
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
                html.Div(
                    html.Span(children=f"{col}: "),
                    html.Span(partnerProfile[col])
                ) 
                for col in partnerFeatures
            ])
        ])
    ])
])

