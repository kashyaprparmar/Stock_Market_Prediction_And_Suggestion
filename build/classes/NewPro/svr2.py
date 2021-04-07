# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:58:44 2019

@author: Acer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('f:/django/aapl_2_data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:,6].values

X_list = np.array(X).tolist()
i=0
while i<len(X_list):
    X_list[i] = str(X_list[i])
    i = i+1
i=0
while i<len(X_list):            #Joining the dates(in form of string to a big string number)
    X_list[i] = ''.join(e for e in X_list[i] if e.isalnum())
    i=i+1
i=0
while i<len(X_list):
    X_l = int(X_list[i])            #typecasting
    X_l = X_l / 100000.0
    X_list[i] = X_l
    i = i+1
X = np.array(X_list)

new_len = len(X_list)

i=0
while i<new_len:
    X_list[i]=i+1;
    i = i+1

pre=3.0
p = float(pre)
m = max(X_list) + p
X = np.array(X_list)
X = X.reshape(-1,1)


# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel ='rbf')
regressor.fit(X,y)
# Predicting a new result
regressor.predict(np.array([6.5]).reshape(1, 1))
#print(regressor.predict(np.array([6.5]).reshape(1, 1)))

#y_pred = regressor.predict(6.5)                #Ctrl+Enter

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
