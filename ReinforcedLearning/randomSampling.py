"""
@author: Suna Ayhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('Ads.csv')
import random
N = 10000
d=10
toplam = 0
secilenler = []
for n in range (0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    toplam = toplam + odul
     
    
plt.hist(secilenler)
plt.show()  