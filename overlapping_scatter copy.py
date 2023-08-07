import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

#df = pd.read_csv('finance-charts-apple.csv')
df = px.data.iris()

fig = go.Figure()

fig.add_trace(go.Mesh3d(
    # 8 vertices of a cube
    x=[0, 0, 1, 1, 0, 0, 1, 1],
    y=[0, 1, 1, 0, 0, 1, 1, 0],
    z=[0, 0, 0, 0, 1, 1, 1, 1],
    # i, j and k give the vertices of triangles
    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
    name='y', opacity=0.20
))

fig.add_trace(go.Scatter3d(x=df['petal_length'],
              y=df['sepal_length'], z=df['sepal_width']))



if __name__ == '__main__':
    fig.show()
