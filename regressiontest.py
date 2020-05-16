#!/usr/bin/env python
# coding: utf-8

# In[30]:


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
x=np.genfromtxt('train_X.csv',delimiter=',', skip_header=1)
y=np.genfromtxt('train_Y.csv',delimiter=',')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
knn = KNeighborsRegressor(n_neighbors=12)
knn.fit(x_train, y_train)
y_pred1 = knn.predict(x_test)
dec = DecisionTreeRegressor(max_depth=5)
dec.fit(x_train, y_train)
y_pred2 = dec.predict(x_test)
mse1 = metrics.mean_squared_error(y_test, y_pred1)
mse2 = metrics.mean_squared_error(y_test, y_pred2)
print(mse1)
print(mse2)

