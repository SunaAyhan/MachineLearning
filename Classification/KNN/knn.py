# -*- coding: utf-8 -*-
"""
@author: Suna Ayhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4].values #bagimsiz degiskenler
y=veriler.iloc[:,4:].values #bagimli degisken

#egitim ve test bolmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=0 )

#egit ve uygula (fit / transform)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#K Neighbors Method
from sklearn.neighbors import KNeighborsClassifier
#komşu sayısı doğru bulma oranında önemlidir
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski') # komşu = 1, uzaklık bulma yöntemi = minkowski
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


#doğruluk oranı
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)