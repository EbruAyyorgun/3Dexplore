import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px


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


df = px.data.iris()

x_col = 'petal_length'
y_col = 'sepal_length'
z_col = 'sepal_width'

boxes = ['petal_width', 'petal_length']

box_plots = go.Figure()
for col in boxes:
  box_plots.add_trace(go.Box(y=df[col], name=col))

#box_plots.add_trace(go.Box(y=df['petal_width']))
#box_plots.add_trace(go.Box(y=df['petal_length']))

fig = go.Figure()

fig.add_trace(go.Mesh3d( name= "cube",
    # 8 vertices of a cube
    x=[0, 0, 1, 1, 0, 0, 1, 1],
    y=[0, 1, 1, 0, 0, 1, 1, 0],
    z=[0, 0, 0, 0, 1, 1, 1, 1],
    # i, j and k give the vertices of triangles
    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], opacity=0.20
))

fig.add_trace(go.Scatter3d(name = "scatter", x=df[x_col],
              y=df[y_col], z=df[z_col]))

fig.update_xaxes(range=[1.5, 4.5])
fig.update_yaxes(range=[3, 9])
fig.update_yaxes(range=[3, 9])

# Step 3. Create a plotly figure


# Step 4. Create a Dash layout
app.layout = html.Div([
    html.Div(children=[dcc.Graph(id='plot', figure=fig, style={'display': 'inline-block'}),
                       dcc.Graph(id='box', figure=box_plots, style={'display': 'inline-block'})]),
    html.H4(f'X axis: {x_col}'),
    dcc.Slider(0, 9, marks=None, value=0, id='xslider',
                    tooltip={"placement": "bottom", "always_visible": True}),
    html.H4(f'Y axis: {y_col}'),
    dcc.Slider(0, 9, marks=None, value=0, id='yslider',
                    tooltip={"placement": "bottom", "always_visible": True}),
    html.H4(f'Z axis: {z_col}'),
    dcc.Slider(0, 9, marks=None, value=0, id='zslider',
                    tooltip={"placement": "bottom", "always_visible": True}),
    html.H4('Box Size:'),
    dcc.Slider(0, 9, marks=None, value=0, id='size',
                    tooltip={"placement": "bottom", "always_visible": True}),
    dcc.Dropdown(list(df.columns), id="box_dropdown", multi=True)
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
    
    newX = [0 + X, 0 + X, 1 + X + size, 1 + X + size, 0 + X, 0 + X, 1 + X + size, 1 + X  + size]
    newY = [0 + Y, 1 + Y + size, 1 + Y + size, 0 + Y, 0 + Y, 1 + Y + size, 1 + Y + size, 0 + Y]
    newZ = [0 + Z, 0 + Z, 0 + Z, 0 + Z, 1 + Z + size, 1 + Z + size, 1 + Z + size, 1 + Z + size]

    

    fig.update_traces(x=newX, y=newY, z=newZ, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                      j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                      k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], selector=dict(name="cube"))

    fig.update_xaxes(range=[1.5, 4.5])
    fig.update_yaxes(range=[3, 9])
    fig.update_yaxes(range=[3, 9])


    return fig


@app.callback(Output('box', 'figure'), [Input('xslider', 'value'),
              Input('yslider', 'value'),
              Input('zslider', 'value'),
              Input('size', 'value'),
              Input('box_dropdown', 'value')])
def update_box(X, Y, Z, size, values):
    #df_filtered = df['petal_length'].between(
    #    X-1, X, inclusive=True)

    # add or delete any boxes based on the dropdown selections

    # filter the dataframe based on the selection
    df_filtered = df[(((df[x_col] >= X) & (df[x_col] <= X+size+1)) 
                      & ((df[y_col] >= Y) & (df[y_col] <= Y+size+1))
                      & ((df[z_col] >= Z) & (df[z_col] <= Z+size+1)))]

    # make sure this applies to all box plot traces in the figure
    for col in boxes:
        box_plots.update_traces(
            y=df_filtered[col], selector=dict(name=col))
    return box_plots

# create or new box plots based on selections from the drop down
def generate_box():
    return



# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server()