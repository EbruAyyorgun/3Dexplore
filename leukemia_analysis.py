import plotly.graph_objs as go
import plotly.express as px
import umap
from umap import UMAP
import pandas as pd
import math
import numpy as np

with open('Leukemia_GSE9476.csv', 'r') as f:
  df = pd.read_csv(f)
#X = df.drop(['UVA_sample_id', 'Run'], axis=1)
pd.set_option('display.max_rows', None)

df['type'] = pd.factorize(df['type'])[0] + 1

reducer = UMAP(n_neighbors=5, min_dist=0.05, n_components=3,
               init='random', random_state=0)
embedding = reducer.fit_transform(df)
embedding_df = pd.DataFrame(embedding)

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    name="scatter", x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], marker = dict(color=df['type'])))
#fig.add_trace(go.Scatter3d(
#    name="scatter", x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], marker = dict(color=df['type'])))

if __name__ == '__main__':
    fig.show()
