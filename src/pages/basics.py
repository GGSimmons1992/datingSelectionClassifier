from dash import Dash, html, dcc
import plotly.express as px

Dash.register_page(__name__)

#css styles
fullPage = {
    "height": "95%",
    "width": "95%"
}

col4 = {
    "width": "32%"
}

col6 = {
    "width":"48%"
}

middleAndCenter = {
    "text-align":"center",
    "vertical-align":"middle"
}

fitContent = {
    "display":"inline-block",
    "height":"fit-content",
    "width":"fit-content"
}

hidden = {
    "display":"hidden"
}

nostyle = {}

layout = html.Div(children=[
    html.H2(children=__name__),
    html.Div(style=fullPage,children=[
        html.H3(children="Background Info"),
        html.Div(children=[
            html.Div(style=col4,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="age-dropdown")
                    ])
                ]),
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="race-dropdown")
                    ])
                ]),
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="age-dropdown")
                    ])
                ])
            ]),
            html.Div(style=col4,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="underga-dropdown")
                    ])
                ]),
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="field-dropdown")
                    ])
                ]),
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="careerCode-dropdown")
                    ])
                ])
            ]),
            html.Div(style=col4,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Input(type="text",id="from-input")
                    ])  
                ]),
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Input(type="text",id="zipcode-input")
                    ])
                ]),
            ]),
        ]),
        html.H3(children="History and Intent"),
        html.Div(children=[
            html.Div(style=col4,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="goal-dropdown")
                    ])  
                ])
            ]),
            html.Div(style=col4,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="date-dropdown")
                    ])  
                ])
            ]),
            html.Div(style=col4,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="goout-dropdown")
                    ])  
                ])
            ]),
        ]),
        html.H3(children="Preferences Filters"),
        html.Div(children=[
            html.Div(style=col6,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="success-dropdown")
                    ])  
                ])
            ]),
            html.Div(style=col6,children=[
                html.Div(style=middleAndCenter,children=[
                    html.Div(style=fitContent,children=[
                        html.Label(),
                        dcc.Dropdown(id="preference-dropdown")
                    ])  
                ])
            ]),
        ]),   
    ]),
    html.Button("Next",id="next"),
    html.Button("Match up",style=hidden,id="matchup")
])