import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import umap
from umap import UMAP

import dash_bootstrap_components as dbc





# Launch the application
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Step 2. Import the dataset
#df = pd.read_csv('finance-charts-apple.csv')


with open('/mnt/c/Users/Ebru-as-user/Downloads/Leukemia_GSE9476.csv', 'r') as f:
  df = pd.read_csv(f)
# saving the string version of the type column to use for labelling
df_type_copy = df['type']

df['type'] = pd.factorize(df['type'])[0] + 1

# UMAP projection of data
reducer = UMAP(n_neighbors=3, min_dist=0.6, n_components=3,
               init='random', random_state=0)
embedding = reducer.fit_transform(df)


# Labels for axes of each plot
x_col = 'UMAP X'
y_col = 'UMAP Y'
z_col = 'UMAP Z'

x_col2 = 'FLT3'
y_col2 = 'NPM1'
z_col2 = 'IDH1'

# 3D arrays for the first and second plot
plot1_axes = embedding

plot2_axes = np.array(
    df[['206674_at', '221923_s_at', '201193_at']].values.tolist())

#boxes = ['214983_at', '206082_at', '210794_s_at', '203591_s_at']

# default columns to use for the box-and-whisker plot if the user doesn't select anything
boxes = ['205780_at', '211725_s_at', '208478_s_at', '203728_at']


# storing the max and mins of each dataset for later use
max_x1 = max(plot1_axes[:, 0])
min_x1 = min(plot1_axes[:, 0])

max_y1 = max(plot1_axes[:, 1])
min_y1 = min(plot1_axes[:, 1])

max_z1 = max(plot1_axes[:, 2])
min_z1 = min(plot1_axes[:, 2])

max_x2 = max(plot2_axes[:, 0])
min_x2 = min(plot2_axes[:, 0])

max_y2 = max(plot2_axes[:, 1])
min_y2 = min(plot2_axes[:, 1])

max_z2 = max(plot2_axes[:, 2])
min_z2 = min(plot2_axes[:, 2])

box_plots = go.Figure()
#for col in boxes:
#  box_plots.add_trace(go.Box(y=df[col], name=col))

#box_plots.add_trace(go.Box(y=df['petal_width']))
#box_plots.add_trace(go.Box(y=df['petal_length']))

# Creating the figures: each have one selection cube and whatever the plot will be
fig = go.Figure()

fig.add_trace(go.Mesh3d(name="cube",
                        # 8 vertices of a cube
                        x=[0, 0, 1, 1, 0, 0, 1, 1],
                        y=[0, 1, 1, 0, 0, 1, 1, 0],
                        z=[0, 0, 0, 0, 1, 1, 1, 1],
                        # i, j and k give the vertices of triangles
                        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], opacity=0.20
                        ))

fig2 = go.Figure()


fig2.add_trace(go.Mesh3d(name="cube",
                        # 8 vertices of a cube
                         x=[0 + min_x2, 0 + min_x2, 1 + min_x2, 1 + min_x2,
                            0 + min_x2, 0 + min_x2, 1 + min_x2, 1 + min_x2],
                         y=[0 + min_y2, 1 + min_y2, 1 + min_y2, 0 +
                            min_y2, 0 + min_y2, 1 + min_y2, 1 + min_y2, 0 + min_y2],
                         z=[0 + min_z2, 0 + min_z2, 0 + min_z2, 0 + min_z2, 1 +
                            min_z2, 1 + min_z2, 1 + min_z2, 1 + min_z2],
                        # i, j and k give the vertices of triangles
                        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], opacity=0.20
                        ))

#fig.add_trace(go.Scatter3d(
#    name="Bone_Marrow_CD34", x=embedding[:, 0][0:8], y=embedding[:, 1][0:8], z=embedding[:, 2][0:8]))

#fig.add_trace(go.Scatter3d(
#    name="Bone_Marrow", x=embedding[:, 0][9:18], y=embedding[:, 1][9:18], z=embedding[:, 2][9:18]))

#fig.add_trace(go.Scatter3d(
#    name="AML", x=embedding[:, 0][19:44], y=embedding[:, 1][19:44], z=embedding[:, 2][19:44]))

#fig.add_trace(go.Scatter3d(
#    name="PB", x=embedding[:, 0][45:53], y=embedding[:, 1][45:53], z=embedding[:, 2][45:53]))

#fig.add_trace(go.Scatter3d(
#    name="PBSC_CD34", x=embedding[:, 0][54:63], y=embedding[:, 1][54:63], z=embedding[:, 2][54:63]))

fig.add_trace(go.Scatter3d(
    name="selected", x=plot1_axes[:, 0], y=plot1_axes[:, 1], z=plot1_axes[:, 2], customdata=df_type_copy, hovertemplate="Type: %{customdata}", marker=dict(opacity=1, color=df['type'])))

fig.add_trace(go.Scatter3d(
    name="unselected", x=plot1_axes[:, 0], y=plot1_axes[:, 1], z=plot1_axes[:, 2], customdata=df_type_copy, hovertemplate="Type: %{customdata}", marker=dict(opacity=0.5, color=df['type'])))


#fig2.add_trace(go.Scatter3d(
#    name="unselected", x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], customdata=df_type_copy, hovertemplate="Type: %{customdata}", marker=dict(opacity=0.5, color=df['type'])))

#fig2.add_trace(go.Scatter3d(
#    name="selected", x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], customdata=df_type_copy, hovertemplate="Type: %{customdata}", marker=dict(opacity=1, color=df['type'])))

fig2.add_trace(go.Scatter3d(
    name="selected", x=plot2_axes[:, 0], y=plot2_axes[:, 1], z=plot2_axes[:, 2], customdata=df_type_copy, hovertemplate="Type: %{customdata}",  marker=dict(opacity=1, color=df['type'])))

fig2.add_trace(go.Scatter3d(
    name="unselected", x=plot2_axes[:, 0], y=plot2_axes[:, 1], z=plot2_axes[:, 2], customdata=df_type_copy, hovertemplate="Type: %{customdata}",  marker=dict(opacity=0.5, color=df['type'])))



fig.update_layout(showlegend = True)

#fig.update_xaxes(range=[1.5, 4.5])
#fig.update_yaxes(range=[3, 9])
#fig.update_yaxes(range=[3, 9])




# Organizing the app layout
app.layout = html.Div([dbc.Row([html.H1("Dash for 3D Data Exploration", style={'textAlign': 'center', 'color': '#7FDBFF'}),
                                html.H4(f'Select Variable for Propagation', style={
                                        'color': '#7FDBFF'}),
                                dcc.Dropdown(list(df.columns),
                                             id="box_dropdown", multi=True),
                                html.H5("Note: Hover over the points to see their color-coded classifications")]),
                       html.Br(),
                       dbc.Row([
                            dbc.Col([
                                dbc.Row(
                                    html.H2("UMAP Projection of Leukemia gene expression")),
                                html.Br(),
                                dbc.Row(html.Div(children=[html.H4('Select on first graph:', style={
                                        'color': '#7FDBFF', 'display': 'inline-block'}), daq.BooleanSwitch(id='include1', on=True, style={'display': 'inline-block'})])),
                                html.H4('Exploration Box Size:',
                                        style={'color': '#7FDBFF'}),
                                dcc.Slider(1, 15, value=4, marks=None, id='size1',
                                           tooltip={"placement": "bottom", "always_visible": True}),
                                dbc.Row(dcc.Graph(id='plot1', figure=fig)),
                                dbc.Row([
                                    dbc.Col([html.H4(f'X axis: {x_col}', style={'color': '#7FDBFF'}),
                                             dcc.Slider(min_x1, max_x1, marks=None, value=0, id='xslider1',
                                        tooltip={"placement": "bottom", "always_visible": True})]),
                                    dbc.Col([html.H4(f'Y axis: {y_col}', style={'color': '#7FDBFF'}),
                                            dcc.Slider(min_y1, max_y1, marks=None, value=0, id='yslider1',
                                                       tooltip={"placement": "bottom", "always_visible": True})]),
                                    dbc.Col([html.H4(f'Z axis: {z_col}', style={'color': '#7FDBFF'}),
                                            dcc.Slider(min_z1, max_z1, marks=None, value=0, id='zslider1',
                                                       tooltip={"placement": "bottom", "always_visible": True})])
                                ])], style={'border': '3px solid black'}
                            ),
                            dbc.Col([
                                dbc.Row(
                                    html.H2("Expression of top markers according to article below")),
                                html.A(
                                    "Link to Paper", href="https://arupconsult.com/ati/acute-myeloid-leukemia-molecular-genetic-testing"),
                                html.Br(),
                                dbc.Row(html.Div(children=[html.H4('Select on second graph:', style={
                                        'color': '#7FDBFF', 'display': 'inline-block'}), daq.BooleanSwitch(id='include2', on=True, style={'display': 'inline-block'})])),
                                html.H4('Exploration Box Size:',
                                        style={'color': '#7FDBFF'}),
                                dcc.Slider(1, 15, marks=None, value=0, id='size2',
                                           tooltip={"placement": "bottom", "always_visible": True}),
                                dbc.Row(dcc.Graph(id='plot2', figure=fig2)),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4(f'X axis: {x_col2}', style={
                                                'color': '#7FDBFF'}),
                                        dcc.Slider(min_x2, max_x2, marks=None, value=min_x2, id='xslider2',
                                            tooltip={"placement": "bottom", "always_visible": True})
                                    ]),
                                    dbc.Col([html.H4(f'Y axis: {y_col2}', style={'color': '#7FDBFF'}),
                                             dcc.Slider(min_y2, max_y2, marks=None, value=min_y2, id='yslider2',
                                                        tooltip={"placement": "bottom", "always_visible": True})
                                    ]),
                                    dbc.Col([html.H4(f'Z axis: {z_col2}', style={'color': '#7FDBFF'}),
                                             dcc.Slider(min_z2, max_z2, marks=None, value=min_z2, id='zslider2',
                                                        tooltip={"placement": "bottom", "always_visible": True})
                                    ])
                                ])], style={'border': '3px solid black'}
                            ),
                        dbc.Col(dcc.Graph(id='box', figure=box_plots))]
)])


# Step 4. Create a Dash layout

#list(df.columns)


# Step 5. Add callback functions


@app.callback(Output('plot1', 'figure'),
              [Input('xslider1', 'value'),
              Input('yslider1', 'value'),
              Input('zslider1', 'value'),
              Input('size1', 'value'),
              Input('include1', 'on')])
def update_figure1(X, Y, Z, size, on):

    # redraw the cube to the new coordinates
    
    if on == True:
        newX = [0 + X, 0 + X, 1 + X + size, 1 + X +
                size, 0 + X, 0 + X, 1 + X + size, 1 + X + size]
        newY = [0 + Y, 1 + Y + size, 1 + Y + size, 0 +
                Y, 0 + Y, 1 + Y + size, 1 + Y + size, 0 + Y]
        newZ = [0 + Z, 0 + Z, 0 + Z, 0 + Z, 1 + Z +
                size, 1 + Z + size, 1 + Z + size, 1 + Z + size]
        fig.update_traces(x=newX, y=newY, z=newZ, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], opacity = 0.2, selector=dict(name="cube"))
        #fig.update_xaxes(range=[1.5, 4.5])
        #fig.update_yaxes(range=[3, 9])
        #fig.update_yaxes(range=[3, 9])

        fig.update_xaxes(range=[min_x1, max_x1])
        fig.update_yaxes(range=[min_y1, max_y1])
        fig.update_yaxes(range=[min_z1, max_z1])

        em_df = pd.DataFrame(plot1_axes, columns=['0', '1', '2'])
        em_filtered = em_df[(((em_df['0'] >= X) & (em_df['0'] <= X+size))
                            & ((em_df['1'] >= Y) & (em_df['1'] <= Y+size))
                            & ((em_df['2'] >= Z) & (em_df['2'] <= Z+size)))].index.to_list()
        df_filtered = em_df.filter(items=em_filtered, axis=0)
        if not df_filtered.empty:
            fig.update_traces(x=df_filtered.iloc[:, 0], y=df_filtered.iloc[:, 1], z=df_filtered.iloc[:, 2], marker=dict(
                opacity=1), selector=dict(name="selected"))
        else:
            fig.update_traces(marker=dict(opacity=0.5), selector=dict(name="selected"))
    else:
        fig.update_traces(marker=dict(
            opacity=0.5), selector=dict(name="selected"))
        fig.update_traces(opacity=0, selector=dict(name="cube"))

    return fig


@app.callback(Output('plot2', 'figure'),
              [Input('xslider2', 'value'),
              Input('yslider2', 'value'),
              Input('zslider2', 'value'),
              Input('size2', 'value'), 
              Input('include2', 'on')])
def update_figure2(X, Y, Z, size, on):

    if on == True:
        # redraw the cube to the new coordinates

        newX = [0 + X, 0 + X, 1 + X + size, 1 + X +
                size, 0 + X, 0 + X, 1 + X + size, 1 + X + size]
        newY = [0 + Y, 1 + Y + size, 1 + Y + size, 0 +
                Y, 0 + Y, 1 + Y + size, 1 + Y + size, 0 + Y]
        newZ = [0 + Z, 0 + Z, 0 + Z, 0 + Z, 1 + Z +
                size, 1 + Z + size, 1 + Z + size, 1 + Z + size]

        fig2.update_traces(x=newX, y=newY, z=newZ, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], opacity = 0.2, selector=dict(name="cube"))

        fig2.update_xaxes(range=[min_x2, max_x2])
        fig2.update_yaxes(range=[min_y2, max_y2])
        fig2.update_yaxes(range=[min_z2, max_z2])

        em_df = pd.DataFrame(plot2_axes, columns=['0', '1', '2'])
        em_filtered = em_df[(((em_df['0'] >= X) & (em_df['0'] <= X+size))
                            & ((em_df['1'] >= Y) & (em_df['1'] <= Y+size))
                            & ((em_df['2'] >= Z) & (em_df['2'] <= Z+size)))].index.to_list()
        df_filtered = em_df.filter(items=em_filtered, axis=0)
        if not df_filtered.empty:
            fig2.update_traces(x=df_filtered.iloc[:, 0], y=df_filtered.iloc[:, 1], z=df_filtered.iloc[:, 2], marker=dict(
                opacity=1), selector=dict(name="selected"))
        else:
            fig2.update_traces(marker=dict(opacity=0.5), selector=dict(name="selected"))
    else:
        fig2.update_traces(marker=dict(
            opacity=0.5), selector=dict(name="selected"))
        fig2.update_traces(opacity=0, selector=dict(name="cube"))

    return fig2
#df_filtered = df['petal_length'].between(
    #    X-1, X, inclusive=True)

    # add or delete any boxes based on the dropdown selections
    # filter the dataframe based on the selection
    #df_filtered = df[(((df[x_col] >= X) & (df[x_col] <= X+size+1))
    #                  & ((df[y_col] >= Y) & (df[y_col] <= Y+size+1))
    #                  & ((df[z_col] >= Z) & (df[z_col] <= Z+size+1)))]
    # make sure this applies to all box plot traces in the figure
@app.callback(Output('box', 'figure'), [Input('xslider1', 'value'),
              Input('yslider1', 'value'),
              Input('zslider1', 'value'),
              Input('xslider2', 'value'),
              Input('yslider2', 'value'),
              Input('zslider2', 'value'),
              Input('size1', 'value'),
              Input('size2', 'value'),
              Input('include1', 'on'),
              Input('include2', 'on'),
              Input('box_dropdown', 'value')])
def update_box(X1, Y1, Z1, X2, Y2, Z2, size1, size2, on1, on2, values):
    em_df = pd.DataFrame(plot1_axes, columns=['0', '1', '2'])
    em_df2 = pd.DataFrame(plot2_axes, columns=['0', '1', '2'])
    if on1 == True:
        em_filtered1 = em_df[(((em_df['0'] >= X1) & (em_df['0'] <= X1+size1))
                    & ((em_df['1'] >= Y1) & (em_df['1'] <= Y1+size1))
                    & ((em_df['2'] >= Z1) & (em_df['2'] <= Z1+size1)))].index.to_list()
    else:
        em_filtered1 = []
    if on2 == True:
        em_filtered2 = em_df2[(((em_df2['0'] >= X2) & (em_df2['0'] <= X2+size2))
                            & ((em_df2['1'] >= Y2) & (em_df2['1'] <= Y2+size2))
                            & ((em_df2['2'] >= Z2) & (em_df2['2'] <= Z2+size2)))].index.to_list()
    if (on1 == True) and (on2==True):
        em_filtered = np.intersect1d(em_filtered1, em_filtered2)
    elif (on1==True) and (on2 == False):
        em_filtered = em_filtered1
    else:
        em_filtered = em_filtered2
    df_filtered = df.filter(items=em_filtered, axis=0)
    current_boxes = []
    for part in box_plots.data:
        if part.name is not None: current_boxes.append(part.name)
    if values is not None:
        copy_boxes = values
    else:
        copy_boxes = boxes
    for col in copy_boxes:
        if(col not in current_boxes):
            box_plots.add_trace(go.Box(y=df_filtered[col], name=col))
        else:
            box_plots.update_traces(
                y=df_filtered[col], selector=dict(name=col))
    return box_plots



def generate_box():
    return


# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server()
