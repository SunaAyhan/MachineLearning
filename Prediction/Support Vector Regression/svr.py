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


#polinomal oldugunu gor
plt.scatter(X, Y, color='red')


#polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)   #derece tahmini etkiler
x_poly = poly_reg.fit_transform(X) #polinoma donustur
print(x_poly)

#doğrusal regression
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)


#gorsellestirme
plt.scatter(X, Y,color='red')
plt.plot(x, lin_reg.predict(poly_reg.fit_transform(X)))
plt.show()

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcek = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcek = sc2.fit_transform(Y)

#svr
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcek,y_olcek)
plt.scatter(x_olcek, y_olcek, color='red')
plt.plot(x_olcek, svr_reg.predict(x_olcek))

print(svr_reg.predict([[6.6]]))
