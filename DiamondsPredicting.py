import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('P1_diamonds.csv')

df = df.drop(['Unnamed: 0'], axis = 1)

categorical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

for i in range(3):
    new = le.fit_transform(df[categorical_features[i]])
    df[categorical_features[i]] = new

#print(df.head(10).to_string())

x = df[['carat', 'cut', 'color', 'clarity', 'depth', 'x', 'y', 'z']]
y = df[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 25, random_state = 101)

regr = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = 101) 
regr.fit(x_train, y_train.values.ravel())

predictions = regr.predict(x_test)

result = x_test
result['price'] = y_test
result['predictions'] = predictions.tolist()

print(result.to_string())
