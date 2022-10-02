from dash import Dash, html, dcc
import plotly.express as px

Dash.register_page(__name__)

#css styles
fullPage = {
    "height": "95%",
    "width": "95%"
}

layout = html.Div(children=[
    html.H2(children=__name__),
    html.Div(style=fullPage,children=[

    ]),
    html.Button("Next",id="next"),
    html.Button("Match up",style=hidden,id="matchup")
])