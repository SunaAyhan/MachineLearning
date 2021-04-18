# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#verileri al
veriler = pd.read_csv('musteriler.csv')

#verileri dilimle
x=veriler.iloc[:,3:].values

#metodu uygula
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
tahmin = ac.fit_predict(x)
print(tahmin)
plt.scatter(x[tahmin==0,0],x[tahmin==0,1],s=100, c='red')
plt.scatter(x[tahmin==1,0],x[tahmin==1,1],s=100, c='blue')
plt.scatter(x[tahmin==2,0],x[tahmin==2,1],s=100, c='green')
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()
