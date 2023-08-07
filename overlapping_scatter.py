import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

#df = pd.read_csv('finance-charts-apple.csv')
N = 50
fig = go.Figure()
fig.add_trace(go.Mesh3d(x=(60*np.random.randn(N)),
                        y=(25*np.random.randn(N)),
                        z=(40*np.random.randn(N)),
                        opacity=0.5,
                        color='yellow'
                        ))
fig.add_trace(go.Mesh3d(x=(70*np.random.randn(N)),
                        y=(55*np.random.randn(N)),
                        z=(30*np.random.randn(N)),
                        opacity=0.5,
                       color='pink'
                        ))

fig.update_layout(scene=dict(
    xaxis_title='X AXIS TITLE',
    yaxis_title='Y AXIS TITLE',
    zaxis_title='Z AXIS TITLE'),
    width=700,
    margin=dict(r=20, b=10, l=10, t=10))


if __name__ == '__main__':
    fig.show()
