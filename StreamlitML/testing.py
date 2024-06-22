import pandas as pd
df=pd.read_csv("housing.csv")
df=df.sample(1000)
df.to_csv("housing.csv")