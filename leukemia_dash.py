import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import umap
from umap import UMAP


# To-Do:
# Get the box size to change done
# Make a drop-down to select which box-plots to create (max of 10 for now)
# Make the updating real-time
# Find a way to not connect the scatter plots
# Make a drop-down to select the axes of the 3D plot
# Stop the figure from fitting to the plot


# Step 1. Launch the application
app = dash.Dash()

# Step 2. Import the dataset
#df = pd.read_csv('finance-charts-apple.csv')


with open('/mnt/c/Users/Ebru-as-user/Downloads/Leukemia_GSE9476.csv', 'r') as f:
  df = pd.read_csv(f)
df['type'] = pd.factorize(df['type'])[0] + 1

reducer = UMAP(n_neighbors=3, min_dist=0.6, n_components=3,
               init='random', random_state=0)
embedding = reducer.fit_transform(df)

x_col = 'UMAP X'
y_col = 'UMAP Y'
z_col = 'UMAP Z'

#boxes = ['214983_at', '206082_at', '210794_s_at', '203591_s_at']
boxes = ['205780_at', '211725_s_at', '208478_s_at', '203728_at']

box_plots = go.Figure()
for col in boxes:
  box_plots.add_trace(go.Box(y=df[col], name=col))

#box_plots.add_trace(go.Box(y=df['petal_width']))
#box_plots.add_trace(go.Box(y=df['petal_length']))

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
    name="unselected", x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], marker=dict(color="red")))

fig.add_trace(go.Scatter3d(
    name="selected", x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], marker=dict(color="red")))




fig.update_layout(showlegend = True)

#fig.update_xaxes(range=[1.5, 4.5])
#fig.update_yaxes(range=[3, 9])
#fig.update_yaxes(range=[3, 9])

# Step 3. Create a plotly figure
max_x = max(embedding[:, 0])
min_x = min(embedding[:, 0])

max_y = max(embedding[:, 1])
min_y = min(embedding[:, 1])

max_z = max(embedding[:, 2])
min_z = min(embedding[:, 2])


# Step 4. Create a Dash layout
app.layout = html.Div([
    html.H1("Dash for 3D Data Exploration", style={'textAlign': 'center', 'color': '#7FDBFF'}),
    html.H4(f'Select Variable for Propagation', style={'color': '#7FDBFF'}),
    dcc.Dropdown(list(df.columns), id="box_dropdown", multi=True),
    html.H4('Exploration Box Size:', style={'color': '#7FDBFF'}),
    dcc.Slider(1, 15, marks=None, value=0, id='size',
               tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(children=[dcc.Graph(id='plot', figure=fig, style={'display': 'inline-block'}),
                       dcc.Graph(id='box', figure=box_plots, style={'display': 'inline-block'})]),
    html.H4(f'X axis: {x_col}', style={'color': '#7FDBFF'}),
    dcc.Slider(min_x, max_x, marks=None, value=0, id='xslider',
               tooltip={"placement": "bottom", "always_visible": True}),
    html.H4(f'Y axis: {y_col}', style={'color': '#7FDBFF'}),
    dcc.Slider(min_y, max_y, marks=None, value=0, id='yslider',
               tooltip={"placement": "bottom", "always_visible": True}),
    html.H4(f'Z axis: {z_col}', style={'color': '#7FDBFF'}),
    dcc.Slider(min_z, max_z, marks=None, value=0, id='zslider',
               tooltip={"placement": "bottom", "always_visible": True})
])
#list(df.columns)


# Step 5. Add callback functions


@app.callback(Output('plot', 'figure'),
              [Input('xslider', 'value'),
              Input('yslider', 'value'),
              Input('zslider', 'value'),
              Input('size', 'value')])
def update_figure(X, Y, Z, size):

    # redraw the cube to the new coordinates

    newX = [0 + X, 0 + X, 1 + X + size, 1 + X +
            size, 0 + X, 0 + X, 1 + X + size, 1 + X + size]
    newY = [0 + Y, 1 + Y + size, 1 + Y + size, 0 +
            Y, 0 + Y, 1 + Y + size, 1 + Y + size, 0 + Y]
    newZ = [0 + Z, 0 + Z, 0 + Z, 0 + Z, 1 + Z +
            size, 1 + Z + size, 1 + Z + size, 1 + Z + size]

    fig.update_traces(x=newX, y=newY, z=newZ, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                      j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                      k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], selector=dict(name="cube"))

    fig.update_xaxes(range=[1.5, 4.5])
    fig.update_yaxes(range=[3, 9])
    fig.update_yaxes(range=[3, 9])

    em_df = pd.DataFrame(embedding, columns=['0', '1', '2'])
    em_filtered = em_df[(((em_df['0'] >= X) & (em_df['0'] <= X+size))
                         & ((em_df['1'] >= Y) & (em_df['1'] <= Y+size))
                         & ((em_df['2'] >= Z) & (em_df['2'] <= Z+size)))].index.to_list()
    df_filtered = em_df.filter(items=em_filtered, axis=0)
    if not df_filtered.empty:
        fig.update_traces(x=df_filtered.iloc[:, 0], y=df_filtered.iloc[:, 1], z=df_filtered.iloc[:, 2], marker=dict(
            color="blue"), selector=dict(name="selected"))

    return fig

#df_filtered = df['petal_length'].between(
    #    X-1, X, inclusive=True)

    # add or delete any boxes based on the dropdown selections
    # filter the dataframe based on the selection
    #df_filtered = df[(((df[x_col] >= X) & (df[x_col] <= X+size+1))
    #                  & ((df[y_col] >= Y) & (df[y_col] <= Y+size+1))
    #                  & ((df[z_col] >= Z) & (df[z_col] <= Z+size+1)))]
    # make sure this applies to all box plot traces in the figure
@app.callback(Output('box', 'figure'), [Input('xslider', 'value'),
              Input('yslider', 'value'),
              Input('zslider', 'value'),
              Input('size', 'value'),
              Input('box_dropdown', 'value')])
def update_box(X, Y, Z, size, values):
    em_df = pd.DataFrame(embedding, columns=['0', '1', '2'])
    em_filtered = em_df[(((em_df['0'] >= X) & (em_df['0'] <= X+size))
                & ((em_df['1'] >= Y) & (em_df['1'] <= Y+size))
                & ((em_df['2'] >= Z) & (em_df['2'] <= Z+size)))].index.to_list()
    df_filtered = df.filter(items=em_filtered, axis=0)
    for col in boxes:
        box_plots.update_traces(
            y=df_filtered[col], selector=dict(name=col))
    return box_plots

# change the color of the selected points in the scatter


# create or new box plots based on selections from the drop down


def generate_box():
    return


# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server()
