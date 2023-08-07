import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# things to fix: 
# why is it a triangle and not a cube when I want to move the box
# add labels to x, y, and z changes
# make it so the size of the axes don't change as you change the position of the box
# add a box and whisker plot
# add a callback to the box and whisker with the filtered dataset



# Step 1. Launch the application
app = dash.Dash()

# Step 2. Import the dataset
#df = pd.read_csv('finance-charts-apple.csv')

#fig = go.Figure()

#fig.add_trace(go.Scatter3d(x=df['AAPL.Open'],
#              y=df['AAPL.High'], z=df['AAPL.Low']))

#fig.add_trace(go.Mesh3d(
#        # 8 vertices of a cube
#        x=[0, 0, 1, 1, 0, 0, 1, 1],
#        y=[0, 1, 1, 0, 0, 1, 1, 0],
#        z=[0, 0, 0, 0, 1, 1, 1, 1],
#        # i, j and k give the vertices of triangles
#        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
#        name='y', opacity=0.20
#    ))


df = px.data.iris()

x_col = 'petal_length'
y_col = 'sepal_length'
z_col = 'sepal_width'

box1 = 'petal_width'
box2 = ''

box_plots = go.Figure()
box_plots.add_trace(go.Box(y=df['petal_width']))

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
#scatter = px.scatter_3d(x=df.Date, y=df['AAPL.High'], z = df['AAPL.Low'])

#fig = go.Figure(data=[
#    go.Mesh3d(
#        # 8 vertices of a cube
#        x=[0, 0, 1, 1, 0, 0, 1, 1],
#        y=[0, 1, 1, 0, 0, 1, 1, 0],
#        z=[0, 0, 0, 0, 1, 1, 1, 1],
#        # i, j and k give the vertices of triangles
#        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
#        name='y', opacity=0.20
#    )
#])

#fig.update_layout(
#    scene=dict(
#        xaxis=dict(nticks=4, range=[0, 10],),
#        yaxis=dict(nticks=4, range=[0, 10],),
#        zaxis=dict(nticks=4, range=[0, 10],),),
#    width=700,
#    margin=dict(r=20, l=10, b=10, t=10))

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
                    tooltip={"placement": "bottom", "always_visible": True})
                      ])
    

# Step 5. Add callback functions


@app.callback(Output('plot', 'figure'),
              [Input('xslider', 'value'),
              Input('yslider', 'value'),
              Input('zslider', 'value'),
              Input('size', 'value')])
def update_figure(X, Y, Z, size):
    
    # redraw the cube to the new coordinates
    #newX = [0 + X + size, 0 + X + size, 1 +
    #        X + size, 1 + X + size, 0 + X + size, 0 + X + size, 1 + X + size, 1 + X + size]
    #newY = [0 + Y + size, 1 + Y + size, 1 +
    #        Y + size, 0 + Y + size, 0 + Y + size, 1 + Y + size, 1 + Y + size, 0 + Y + size]
    #newZ = [0 + Z + size, 0 + Z + size, 0 +
    #        Z + size, 0 + Z + size, 1 + Z + size, 1 + Z + size, 1 + Z + size, 1 + Z + size]
    #newX = [0 + X , 0 + X, 1 +
    #        X, 1 + X , 0 + X, 0 + X , 1 + X , + X ]
    #newY = [0 + Y , 1 + Y , 1 +
    #        Y , 0 + Y , 0 + Y , 1 + Y , 1 + Y , 0 + Y ]
    #newZ = [0 + Z, 0 + Z, 0 +
    #        Z , 0 + Z , 1 + Z , 1 + Z , 1 + Z , 1 + Z ]
    
    newX = [0 + X, 0 + X, 1 + X, 1 + X, 0 + X, 0 + X, 1 + X, 1 + X]
    newY = [0 + Y, 1 + Y, 1 + Y, 0 + Y, 0 + Y, 1 + Y, 1 + Y, 0 + Y]
    newZ = [0 + Z, 0 + Z, 0 + Z, 0 + Z, 1 + Z, 1 + Z, 1 + Z, 1 + Z]

    #newX = [x + size for x in newX]
    #newX = [y + size for y in newY]
    #newZ = [z + size for z in newZ]
    #fig = go.Figure(data=[
    #    go.Mesh3d(
    #        # 8 vertices of a cube
    #        x=newX,
    #        y=newY,
    #        z=newZ,
    #        # i, j and k give the vertices of triangles
    #        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    #        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    #        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
    #        name='y', opacity=0.20
    #    )
    #])

    fig.update_traces(x=newX, y=newY, z=newZ, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                      j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                      k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], selector=dict(name="cube"))

    fig.update_xaxes(range=[1.5, 4.5])
    fig.update_yaxes(range=[3, 9])
    fig.update_yaxes(range=[3, 9])

    

    #fig.update_layout(
    #    scene=dict(
    #        xaxis=dict(nticks=4, range=[0, 10],),
    #        yaxis=dict(nticks=4, range=[0, 10],),
    #        zaxis=dict(nticks=4, range=[0, 10],),),
    #    width=700,
    #    margin=dict(r=20, l=10, b=10, t=10))

    return fig


@app.callback(Output('box', 'figure'), [Input('xslider', 'value'),
              Input('yslider', 'value'),
              Input('zslider', 'value')])
def generate_box(X, Y, Z):
    #df_filtered = df['petal_length'].between(
    #    X-1, X, inclusive=True)

    df_filtered = df[(((df[x_col] >= X) & (df[x_col] <= X+1)) 
                      & ((df[y_col] >= Y) & (df[y_col] <= Y+1))
                      & ((df[z_col] >= Z) & (df[z_col] <= Z+1)))]
    
    #print(df_filtered)

    box_plots.update_traces(y=df_filtered[box1])
    #box_plots = go.Box(y=df_filtered['petal_width'])
    return box_plots



# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server()