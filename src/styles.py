import numpy as np
#css styles
maxWidth = 95
col6Width = np.floor(maxWidth/2)
col4Width = np.floor(maxWidth/3)
col3Width = np.floor(maxWidth/4)
col2Width = np.floor(maxWidth/6)
col1Width = np.floor(maxWidth/12)
col11Width = maxWidth - col1Width
col10Width = maxWidth - col2Width
col9Width = maxWidth - col3Width
col8Width = maxWidth - col4Width
col7Width = col8Width - col1Width
col5Width = maxWidth - col7Width


col12 = {
    "width": str(maxWidth) + "%",
    "display":"inline-block"
}

col6 = {
    "width":str(col6Width) + "%",
    "display":"inline-block"
}

col4 = {
    "width": str(col4Width) + "%",
    "display":"inline-block"
}

col3 = {
    "width": str(col3Width) + "%",
    "display":"inline-block"
}

col2 = {
    "width": str(col2Width) + "%",
    "display":"inline-block"
}

col1 = {
    "width": str(col1Width) + "%",
    "display":"inline-block"
}

col11 = {
    "width": str(col11Width) + "%",
    "display":"inline-block"
}

col10 = {
    "width": str(col10Width) + "%",
    "display":"inline-block"
}

col9 = {
    "width": str(col9Width) + "%",
    "display":"inline-block"
}

col8 = {
    "width": str(col8Width) + "%",
    "display":"inline-block"
}

col7 = {
    "width": str(col7Width) + "%",
    "display":"inline-block"
}

col5 = {
    "width": str(col5Width) + "%",
    "display":"inline-block"
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

selected = {
    "text-align":"center",
    "vertical-align":"middle",
    "background-color":"deepskyblue"
}

unselected = {
    "text-align":"center",
    "vertical-align":"middle",
    "background-color":"gainsboro"
}

displayHidden = {
    "display":"none"
}

displayBlock = {
    "display":"block"
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

