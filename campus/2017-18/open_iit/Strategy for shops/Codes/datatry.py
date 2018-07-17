# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
df = pd.read_csv('Total.csv')

x = df.iloc[:,1:4].values

y = x[:19,1].reshape(1,-1)
y = np.insert(y,0,'Name of place',0)

names = [x[0,0]]
temp = x[0,0]
for i in range(429):
    if temp != x[i,0]:
        names.append(x[i,0])
        temp = x[i,0]
        
df1 = []
l = []

for i in range(23):
    df1.append(df[df.City == names[i]])
    l.append(len(df[df.City == names[i]]))
