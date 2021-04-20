# -*- coding: utf-8 -*-
"""
@author: Suna Ayhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yukleme
veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4].values #bagimsiz degiskenler
y=veriler.iloc[:,4:].values #bagimli degisken

# egitim ve test bolmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=0 )

#egit ve uygula (fit / transform)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# random forest
from sklearn.ensemble import RandomForestClassifier
rfc =  RandomForestClassifier(n_estimators=10)
rfc.fit(x_train,y_train)


# tahmin ettir
y_pred = rfc.predict(x_test)
print(y_pred)


# tahmin ve gerçek değerler ne kadar uyumlu? diagondakiler dogru sayısını verir
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)