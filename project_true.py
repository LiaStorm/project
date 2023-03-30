from re import template
from turtle import width
from dash import Dash, dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import numpy as np
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

pars_oblasti = pd.read_csv("https://github.com/LiaStorm/project/blob/8025e6ac73a5a8db4fcc213079d9f52eced676cf/pars_oblasti_RF_clean_true.csv", sep=',', on_bad_lines='skip')

# df = px.data.gapminder()
# years = df.year.unique()
# years2 = list(pars_oblasti["published_at_date"]).sort()
continents =pars_oblasti["OblastiRF"].unique()


# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__)
server = app.server

header = html.H4(
    "Проект", className="bg-primary text-white p-2 mb-2 text-center"
)

table1 = html.Div(
    dash_table.DataTable(
        id="table1",
        columns=[{"name": i, "id": i, "deletable": True} for i in pars_oblasti.columns],
        data=pars_oblasti.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table2 = html.Div(
    dash_table.DataTable(
        id="table2",
        columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
        data=df.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
data_des = pars_oblasti.describe()

table55 = html.Div(
    dash_table.DataTable(
        id="table55",
        columns=[{"name": i, "id": i, "deletable": True} for i in data_des.columns],
        data=data_des.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        row_selectable="multi",
    ),
    className="dbc-row-selectable",
)


sres = pars_oblasti.groupby(["OblastiRF"]).agg({
  "salary": "mean",
  "Density": "mean",
  "Count_word": "mean"
}).reset_index()

dff = pars_oblasti.fillna(0)
table3 = html.Div(dcc.Graph(
        figure=px.scatter(
            sres,
            x="Density",
            y="salary",
            title=f"Распределение количесва слов в описании, ЗП и Населения",
            size="Count_word",
            color="OblastiRF",
            log_x=True,
            size_max=60,
            ),
        className="border", id="all_stat",
    )
)
data = pars_oblasti
import pandas as pd
plot = go.Figure(data=[go.Scatter(
    x=data["published_at_date"],
    y=data["salary"],
    mode='markers',)
])
# Add dropdown
plot.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["type", "scatter"],
                    label="Scatter Plot",
                    method="restyle"
                ),
                dict(
                    args=["type", "bar"],
                    label="Bar Chart",
                    method="restyle"
                )
            ]),
            direction="down"
        ),
    ]
)
 
table6 = dcc.Graph(
        figure=plot
        ,
        className="border", id="raspr"
    )
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
r = norm.rvs(pars_oblasti["salary"])

x_norm = pars_oblasti["salary"]

fig = go.Figure()
fig.add_trace(go.Histogram(x=r, histnorm='probability density', name="Гистограмма распределения ЗП"))
fig.update_layout(
    title="Гистограмма распределения ЗП",
    title_x = 0.5,
    legend=dict(x=.5, xanchor="center", orientation="h"),
    margin=dict(l=0, r=0, t=30, b=0))


table7 = dcc.Graph(
        figure=fig,
        className="border",id="hist_salary"
    )

fig2 = go.Figure()
fig2.add_trace(go.Box(y=pars_oblasti["salary"], name='2d6'))
fig2.update_layout(title="Распределения ЗП"
                  )
table8 = dcc.Graph(
        figure=fig2 ,
        className="border", id="mean_box_salary"
    )

table4 = html.Div(
    dash_table.DataTable(
        id="table4",
        columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
        data=df.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        row_selectable="multi",
    ),
    className="dbc-row-selectable",
)

dropdown = html.Div(
    [
        dbc.Label("Список областей"),
        dcc.Dropdown(
            [i for i in set(continents)],
            "все",
            id="indicator",
            clearable=False,
            multi=True
        ),
    ],
    className="mb-4",
)
map_mean_salary = html.Div(children=[
        html.Iframe(id='map1', srcDoc=open("https://github.com/LiaStorm/project/blob/8025e6ac73a5a8db4fcc213079d9f52eced676cf/MeanSalary.html", 'r').read(), width = "100%", height = "600")
], style={'padding': 10, 'flex': 1})
map_count_prof = html.Div(children=[
        html.Iframe(id='map2', srcDoc=open("https://github.com/LiaStorm/project/blob/8025e6ac73a5a8db4fcc213079d9f52eced676cf/CountRoles.html", 'r').read(), width = "100%", height = "600")
], style={'padding': 10, 'flex': 1})
map_oblas_prof = html.Div(children=[
        html.Iframe(id='map3', srcDoc=open("https://github.com/LiaStorm/project/blob/8025e6ac73a5a8db4fcc213079d9f52eced676cf/OblastiRoles.html", 'r').read(), width = "100%", height = "600")
], style={'padding': 10, 'flex': 1})
map_prof = html.Div(children=[
        html.Iframe(id='map4', srcDoc=open("https://github.com/LiaStorm/project/blob/8025e6ac73a5a8db4fcc213079d9f52eced676cf/Roles.html", 'r').read(), width = "100%", height = "600")
], style={'padding': 10, 'flex': 1})

checklist = html.Div(
    [
        dbc.Label("Select Continents"),
        dbc.Checklist(
            id="continents",
            options=["Да"],
            value=continents,
            inline=True,
        ),
    ],
    className="mb-4",
)

slider = html.Div(
    [
        dbc.Label("Select Years"),
        dcc.RangeSlider(
            5,
            id="years",
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            value=[1,6],
            className="p-0",
        ),
    ],
    className="mb-4",
)
theme_colors = [
    "primary",
    "secondary",
    "success",
    "warning",
    "danger",
    "info",
    "light",
    "dark",
    "link",
]
colors = html.Div(
    [dbc.Button(f"{color}", color=f"{color}", size="sm") for color in theme_colors]
)
colors = html.Div(["Theme Colors:", colors], className="mt-2")



sres5 = pars_oblasti.groupby(["professional_roles"]).agg({
  "salary": "mean",
  "Density": "mean",
  "Count_word": "mean"
}).reset_index()
fig5 = px.treemap(sres5, path=['professional_roles'], values='salary',
                  color='Density')
fig5.update_layout(
    title="Распределение профессий по величине ср. ЗП в зависимости от плотности населения",
    title_x = 0.5,
    legend=dict(x=.5, xanchor="center", orientation="h"),
    margin=dict(l=0, r=0, t=30, b=0))

treemap_prof= dcc.Graph(
        figure=fig5,
        className="border"
    )
pars_oblasti_sres = pars_oblasti
pars_oblasti_sres["Counter"]=[ 1 for x in range(len(pars_oblasti_sres.id))]
sres6 = pars_oblasti_sres.groupby(["professional_roles"]).agg({
  "salary": "mean",
  "Density": "mean",
  "Count_word": "mean",
  "Counter": "count"
}).reset_index()
fig7 = px.treemap(sres6, path=['professional_roles'], values='Counter',
                  color='salary')
fig7.update_layout(
    title="Распределения проффесий по частоте и величине ср. зп",
    title_x = 0.5,
    legend=dict(x=.5, xanchor="center", orientation="h"),
    margin=dict(l=0, r=0, t=30, b=0))
treemap_prof_count= dcc.Graph(
        figure=fig7,
        className="border"
    ) 

pars_oblasti_sres = pars_oblasti
pars_oblasti_sres["Counter"]=[ 1 for x in range(len(pars_oblasti_sres.id))]
sres7 = pars_oblasti_sres.groupby(["employer"]).agg({
  "salary": "mean",
  "Density": "mean",
  "Count_word": "mean",
  "Counter": "count"
}).reset_index()
sres7_otr = sres7.where(sres7["Counter"]>1).dropna()
fig9 = px.treemap(sres7_otr, path=['employer'], values='Counter',
                  color='salary')
fig9.update_layout(
    title="Распределения проффесий по частоте и величине ср. зп",
    title_x = 0.5,
    legend=dict(x=.5, xanchor="center", orientation="h"),
    margin=dict(l=0, r=0, t=30, b=0))
treemap_emploer= dcc.Graph(
        figure=fig9,
        className="border"
    )
fig10 = px.treemap(sres7_otr, path=['employer'], values='salary',
                  color='Density')
fig10.update_layout(
    title="Распределения проффесий по частоте и величине ср. зп",
    title_x = 0.5,
    legend=dict(x=.5, xanchor="center", orientation="h"),
    margin=dict(l=0, r=0, t=30, b=0))
treemap_emploer_another= dcc.Graph(
        figure=fig10,
        className="border"
    )
# , label="Аналитика по Зарплате"
import string
text_op= pars_oblasti["snippet"].dropna(0)
spec_chars = string.punctuation + '\n\xa0«»\t—…'
text_punct_list = []
for text in text_op:
  if text !=0:
    text_punct = "".join([ch for ch in text if ch not in spec_chars])
    text_punct_list.append(text_punct)
all_text = ""
for text in text_punct_list:
  all_text += text


import nltk
# nltk.download('punkt')
from nltk import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist

text_tokens = word_tokenize(all_text)
russian_stopwords = stopwords.words("russian")
text = nltk.Text(text_tokens)
fdist_sw  = [word for word in text_tokens if not word in russian_stopwords]
fdist_without_sw = FreqDist(fdist_sw)

from wordcloud import WordCloud
import base64
from io import BytesIO
import matplotlib.pyplot as plt


from PIL import Image, ImageDraw
img = Image.open("https://github.com/LiaStorm/project/blob/8025e6ac73a5a8db4fcc213079d9f52eced676cf/freqDist.png")

with BytesIO() as buffer:
    img.save(buffer, 'png')
    img22 = base64.b64encode(buffer.getvalue()).decode()

Cloud_raspr = html.Div(children=[
                    html.Img(src="data:image/png;base64," + img22, className = 'center')
                ],className="dbc-row-selectable", style={
                        "display": "inline-block",
                        "width": "70%",
                        "margin-left": "20px",
                        "verticalAlign": "top"
                    }
        )  


from wordcloud import WordCloud, STOPWORDS

text_raw = " ".join(fdist_sw) 

wordCloud = WordCloud(width = 1000, height = 700).generate(text_raw)

wc_img = wordCloud.to_image()
with BytesIO() as buffer:
    wc_img.save(buffer, 'png')
    img2 = base64.b64encode(buffer.getvalue()).decode()


Cloud = html.Div(children=[
                    html.Img(src="data:image/png;base64," + img2)
                ],className="dbc-row-selectable", style={
                        "display": "inline-block",
                        "width": "80%",
                        "margin-left": "20px",
                        "verticalAlign": "top"
                    }
        )    

pars_oblasti_for_corr= pars_oblasti.drop(labels = ["Unnamed: 0","id","working_days","working_time_intervals","working_time_modes","Counter"], axis = 1)
pars_oblasti_corr = pars_oblasti_for_corr.corr()
fig15 = go.Figure()
fig15.add_trace(
    go.Heatmap(
        x = pars_oblasti_corr.columns,
        y = pars_oblasti_corr.index,
        z = np.array(pars_oblasti_corr),
        text=pars_oblasti_corr.values,
        texttemplate='%{text:.2f}'
    )
)

Corr= dcc.Graph(
        figure=fig15,
        className="border"
    )
f = pars_oblasti["salary"].mean()
ff = pars_oblasti["professional_roles"].count()
fff = pars_oblasti["Count_word"].mean()
ffff = pars_oblasti["Density"].mean()
one = dbc.Card(
        [
            html.H2(f"{f:.2f}", className="card-title"),
            html.P("Средняя ЗП", className="card-text"),
        ],
        body=True,
        color="light",
    )

two = dbc.Card(
        [
            html.H2(f"{ff:.2f}", className="card-title"),
            html.P("Количество вакансий", className="card-text"),
        ],
        body=True,
        color="dark",
    )

tri = dbc.Card(
        [
            html.H2(f"{fff:.2f}", className="card-title"),
            html.P("Среднее кол-во слов в описании", className="card-text"),
        ],
        body=True,
        color="green",
    )

chet= dbc.Card(
        [
            html.H2(f"{ffff:.2f}", className="card-title"),
            html.P("Среднее плотность населения", className="card-text"),
        ],
        body=True,
        color="orange",
    )

tab1 = dbc.Tab([map_mean_salary], label="Карта средней зп")
tab2 = dbc.Tab([table6], label="Распределение по дате")
tab3 = dbc.Tab([table7], label="Распределение")
tab4 = dbc.Tab([table8], label="Бокс")
tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3, tab4]))

tab11 = dbc.Tab([map_count_prof], label="Карта вакансий по регионам")
tab12 = dbc.Tab([map_oblas_prof], label="Карта выбор региона")
tab13 = dbc.Tab([map_prof], label="Карта выбор вакансии")
tab14 = dbc.Tab([treemap_prof], label="Самые высоко опплачиваемые")
tab15 = dbc.Tab([treemap_prof_count], label="Самые распрастраненные")
tab1s = dbc.Card(dbc.Tabs([tab11, tab12, tab13, tab14, tab15]))

tab21 = dbc.Tab([treemap_emploer], label="Самые распрастраненные")
tab22 = dbc.Tab([treemap_emploer_another], label="Самые высоко опплачиваемые")
tab2s = dbc.Card(dbc.Tabs([tab21, tab22]))

tab31 = dbc.Tab([Cloud], label="Облако требований")
tab32 = dbc.Tab([Cloud_raspr], label="Распределение слов")
tab3s = dbc.Card(dbc.Tabs([tab31, tab32]))

tab41 = dbc.Tab([Corr], label="Корреляция")
tab42 = dbc.Tab([table3], label="Распределение количесва слов в описании, ЗП и Населения")
tab43 = dbc.Tab([table55], label="Статистика", className="p-4")
tab45 = dbc.Tab([table1], label="Данные", className="p-4")
tab4s = dbc.Card(dbc.Tabs([tab41, tab42, tab43, tab45]))

cont_them= dbc.Card([ThemeChangerAIO(aio_id="theme"), colors])

controls = dbc.Card(
    [cont_them, dropdown, one, two, tri, chet],
    body=True,
)
header2 = html.H4(
    "Аналитика ЗП", className="bg-primary text-white p-2 mb-2 text-center"
)
header3 = html.H4(
    "Аналитика вакансий", className="bg-primary text-white p-2 mb-2 text-center"
)
header4 = html.H4(
    "Аналитика работодателей", className="bg-primary text-white p-2 mb-2 text-center"
)
header5 = html.H4(
    "Статистика и данные", className="bg-primary text-white p-2 mb-2 text-center"
)
header6 = html.H4(
    "Аналитика требований вакансий", className="bg-primary text-white p-2 mb-2 text-center"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        controls,
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # When running this app locally, un-comment this line:
                    ],
                    width=3,
        style={
                        "display": "inline-block",
                        "text-align": "center"
                        
                    }
                ),
                dbc.Col([header2, tabs ], width=9),
                dbc.Col(
                    [
                        header3, tab1s, header4, tab2s, header6,tab3s, header5, tab4s
                        
                    ],
                    style={
                                        "display": "inline-block",
                                        "text-align": "center"
                                    },width={"size": 10, "offset": 1}
                ),
            ]
        ),
    ],
    fluid=True,
    className="dbc",
)


@callback(
    Output("table", "data"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_line_chart(theme):

    dff = pars_oblasti
    data = dff.to_dict("records")
    return data

#     fig = px.line(
#         dff,
#         x=dff["published_at_date"],
#         y=dff["salary"],
#         color="continent",
#         template=template_from_url(theme),
#     )

#     fig_scatter = px.scatter(
#         dff,
#         x=dff["published_at_date"],
#         y=dff["salary"],
#         size="Density",
#         color="continent",
#         log_x=True,
#         size_max=60,
#         template=template_from_url(theme),
#         title="Gapminder %s: %s theme" % (template_from_url(theme)),
#     )

#     return fig, fig_scatter, data
        

if __name__ == "__main__":
    app.run_server(debug=True)
