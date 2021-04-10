# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

ulke = veriler.iloc[:,0:1].values
print(ulke)
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
c = veriler.iloc[:,-1:].values
print(c)

le = preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)
s=pd.concat([sonuc,sonuc2])


















