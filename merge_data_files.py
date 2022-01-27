import pandas as pd

X_data = pd.read_csv("data/X_data.csv", header=None)
X_data.columns = ["Feature "+str(i+1) for i in range(100)]

y_data = pd.read_csv("data/y_data.csv", header=None)
y_data.columns = ["Target"]

full_data = pd.concat([X_data, y_data], axis=1)
full_data.to_csv("data/full_data.csv", index=False)