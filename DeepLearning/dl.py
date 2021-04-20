# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#verileri oku
veriler = pd.read_csv('Churn_Modelling.csv')

# verileri bağımlı ve bağımsız değişken olarak ayır
x=veriler.iloc[:,3:13].values
y=veriler.iloc[:,13]

#encoder: verilerden numerik olmayanları numerik yap
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x[:,1] = le.fit_transform(x[:,1]) #ulke kolonu

le2 = preprocessing.LabelEncoder()
x[:,2] = le.fit_transform(x[:,2]) #cinsiyet kolonu

# verileri numerik yapmak (0 - 1) 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([('ohe', OneHotEncoder(dtype=float),[1])]
                        ,remainder="passthrough")
x=ohe.fit_transform(x)
x=x[:,1:]


# egitim ve test bolmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=0 )

#egit ve uygula (fit / transform)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()               #verileri ölçeklendir
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#yapay sinir ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

#yapay sinir ağını oluştur
classifier = Sequential()
classifier.add(Dense(6,  activation='relu' , input_dim = 11 )) # (gizli katmanda kaç tane nöron olacak - 6) (initial değeri 0'a yakın olmalı) (dim = giriş katmanı nöron sayımız yani kolon sayısı - 11)
classifier.add(Dense(6,  activation='relu' )) # 2. gizli katmanı ekle
classifier.add(Dense(1,  activation='sigmoid' )) # çıkış katmanını ekle (sadece 1 çıkış var)

# nöronları compile et
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
 # fit ve transform
classifier.fit(x_train, y_train,epochs=50) # epochs -> kaç aşamada öğrenecek? geri yayılım
y_pred = classifier.predict(x_test)

y_pred = (y_pred>0.5) #müşterinin bırakma oranı

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


