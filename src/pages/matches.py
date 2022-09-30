from dash import Dash, html, dcc
import plotly.express as px

Dash.register_page(__name__)

layout = html.Div(children=[
    html.H1(children='Ensemble Dash!'),
    html.H2(children=__name__),
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Link(html.Button("Submit Profile Edit"), href="/matches", refresh=True)

])