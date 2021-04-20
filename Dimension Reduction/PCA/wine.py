# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:01:20 2021

@author: Suna Ayhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#verileri oku
veriler = pd.read_csv('wine.csv')

# verileri bağımlı ve bağımsız değişken olarak ayır
x=veriler.iloc[:,0:13].values
y=veriler.iloc[:,13].values

# egitim ve test bolmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=0 )

#egit ve uygula (fit / transform)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()               #verileri ölçeklendir
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

# pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2, y_train)

#tahminler
y_pred = classifier.predict(x_test)
y_pred2 = classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix

# actual / pca olmadan çıkan sonuç
print('pca yok')
cm =  confusion_matrix(y_test, y_pred)
print(cm)

# actual / pca'dan sonraki sonuç
print('pca var')
cm2 =  confusion_matrix(y_test, y_pred2)
print(cm2)

# pca sonrası / öncesi
print('pcasiz ve pcali')
cm3 =  confusion_matrix(y_pred, y_pred2)
print(cm3)
