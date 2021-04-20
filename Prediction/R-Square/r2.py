# -*- coding: utf-8 -*-
"""
@author: Suna Ayhan
"""
import matplotlib.pyplot as plt
import pandas as pd

#veriyi yukle
veriler = pd.read_csv('maaslar.csv') 


#veriyi ayir (dilimleme/slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy array donusumu
X = x.values
Y = y.values




from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())
plt.scatter(X,Y,color='red')
plt.plot(X, rf_reg.predict(X))

#tahmin
print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')

#r square
from sklearn.metrics import r2_score
print(r2_score(Y, rf_reg.predict(X)))
