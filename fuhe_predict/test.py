import pandas as pd
dataset = pd.read_csv(r"C:\Users\42910\fuhe_predict\xgbmodel\dataset.csv")
dataset = pd.Series(pd.to_numeric(dataset.values[:,1]),index = pd.to_datetime(dataset.values[:,0]))
print(dataset)

