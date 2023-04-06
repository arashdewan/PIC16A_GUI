#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import tree

url = 'https://raw.githubusercontent.com/PhilChodrow/PIC16A/master/datasets/palmer_penguins.csv'
penguins = pd.read_csv(url)

penguins['Species'] = penguins['Species'].str.split().str.get(0)
penguins.groupby(['Island', 'Species'])[['Body Mass (g)', 'Culmen Length (mm)']].aggregate(np.mean)
train, test = train_test_split(penguins, test_size = 0.5)
def prep_penguins(data_df):

    df = data_df.copy()
    le = preprocessing.LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    le = preprocessing.LabelEncoder()
    df['Island'] = le.fit_transform(df['Island'])
    df = df[['Species', 'Island', 'Body Mass (g)', 'Culmen Length (mm)']]
    df = df.dropna()
    X = df.drop(['Species'], axis = 1)
    y = df['Species']
    
    return(X, y)

X_train, y_train = prep_penguins(train)
X_test, y_test   = prep_penguins(test)

# make the model
T = tree.DecisionTreeClassifier(max_depth = 5)
T.fit(X_train, y_train)
T.score(X_train, y_train), T.score(X_test, y_test)



#Using pickle to save a state and open it in a new Python session for the GUI

import pickle

pickle.dump(T, open("model.p", "wb"))
T2 = pickle.load(open("model.p", "rb"))
T2.score(X_test, y_test)


# In[ ]:




