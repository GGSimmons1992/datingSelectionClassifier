import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import styles
import preprocess

featureSelectOptions = preprocess.featureSelectOptions
selectedValue = preprocess.selectedValue
descriptionDictionary = preprocess.descriptionDictionary
selectedMatch = preprocess.selectedMatch
candidateProfile = preprocess.candidateProfile
candidateFeatures = preprocess.candidateFeatures
partnerProfile = preprocess.partnerProfile
partnerFeatures = preprocess.partnerFeatures

dash.register_page(__name__,path="/")

col12 = styles.col12
col6 = styles.col6
col4 = styles.col4
col3 = styles.col3
fitContent = styles.fitContent
displayInlineBlock = styles.displayInlineBlock

featureSelect = dcc.Dropdown(
    id="featureSelect",
    options=featureSelectOptions,
    value=[]
)

featureNumber = html.Div(id='featureNumber',style=displayInlineBlock,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Input(id="featureNumberInput",type="number")
])

featureDropdown = html.Div(id='featureNumber',style=displayInlineBlock,children=[
    html.Span(id="featureNumberLabel"),
    dcc.Dropdown(id='featureDropdownInput')
])

featureQuestion = html.Div(id='featureQuestion',style=displayInlineBlock,children=[
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
            html.H3(children="What do you want to examine"),
            featureSelect
        ]),
        html.H3("Calculated Values: Edit profile traits to change these"),
        html.Div(style=fitContent),
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

