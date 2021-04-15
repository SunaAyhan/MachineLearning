# -*- coding: utf-8 -*-

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



#decision tree
from  sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X, r_dt.predict(X))

#tahmin
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))