from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from os.path import exists
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import styles
import preprocess

modelDescriptionDictionary = preprocess.modelDescriptionDictionary
allEstimatorTuples = preprocess.allEstimatorTuples
matrixDictionary = preprocess.matrixDictionary
metricsTable = preprocess.metricsTable
significantFeaturesDictionary = preprocess.significantFeaturesDictionary


dash.register_page(__name__)

col12 = styles.col12
col6 = styles.col6
displayInlineBlock = styles.displayInlineBlock

modelInfoList = []
for modelInfo in allEstimatorTuples:
    if modelInfo[0] == "Ensemble" or "knn" in modelInfo[0]:
        modInfoLayout = html.Div(style=displayInlineBlock,id=modelInfo[0] + "Info",children=[
            html.Div(children=[
            html.Div(style=col6,children=html.H3(children=modelInfo[0])),
            html.Div(style=col6,children=html.P(children=modelDescriptionDictionary[modelInfo[0]]))
            ]),
            html.Div(children=[
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Matrix",figure=matrixDictionary[modelInfo[0]])
                ]),
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Metrics",
                    figure = go.Figure(go.Bar(
                        x=list(metricsTable[modelInfo[0]].values()), 
                        y=list(metricsTable[modelInfo[0]].keys()),
                        orientation='h',
                        hovertext=list(metricsTable[modelInfo[0]].values()))))
                ])
            ])
        ])
    else:
        modInfoLayout = html.Div(style=displayInlineBlock,id=modelInfo[0] + "Info",children=[
            html.Div(children=[
            html.Div(style=col6,children=html.H3(children=modelInfo[0])),
            html.Div(style=col6,children=html.P(children=modelDescriptionDictionary[modelInfo[0]]))
            ]),
            html.Div(children=[
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Matrix",figure=matrixDictionary[modelInfo[0]])
                ]),
                html.Div(style=col6,children=[
                    dcc.Graph(id=str(modelInfo[0]) + "Metrics",
                    figure = go.Figure(go.Bar(
                        x=list(metricsTable[modelInfo[0]].values()), 
                        y=list(metricsTable[modelInfo[0]].keys()),
                        orientation='h',
                        hovertext=list(metricsTable[modelInfo[0]].values()))))
                ])
            ]),
            html.Div(children=[
                html.H4("Top 10 deciding features:"),
                html.Ol(children=[
                    html.Li(children=feat) for feat in (significantFeaturesDictionary[modelInfo[0]])
                ])
            ])
        ])
    modelInfoList.append(modInfoLayout)

layout = html.Div(style=col12,children=modelInfoList)