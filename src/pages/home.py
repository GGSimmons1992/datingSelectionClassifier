from dash import Dash, html, dcc
import plotly.express as px

Dash.register_page(__name__,path="/")

#css styles
welcome = {
    "height": "95%",
    "width": "95%",
    "text-align":"center",
    "vertical-align":"middle",
    "font-size":"48px"
}

layout = html.Div(children=[
    html.H2(children=__name__),
    html.Div(style=welcome,children="Welcome to the Ensemble Dash Dating Classifier!"),
    html.Button("Next",style={"float":"right","padding-right":"25px"},id="next"),
    html.Button("Match up",style={"float":"right","padding-right":"25px"},id="matchup")
])