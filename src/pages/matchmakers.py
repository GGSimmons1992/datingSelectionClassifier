from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import styles
import ensembleDash

datingTrain = ensembleDash.datingTrain
featureSelectColumns = ensembleDash.featureSelectColumns

Dash.register_page(__name__)

featureSelct = dcc.Dropdown(
options=[col for col in datingTrain.columns]
)

featureNumber = html.Div(children=[
    html.label(),
    dcc.Input(type="number")
])

featureDropdown = dcc.Dropdown()

layout = html.Div(children=[])