import plotly.express as px

df = px.data.iris()


if __name__ == "__main__":
    print(list(df.columns))
