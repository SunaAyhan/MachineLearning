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



