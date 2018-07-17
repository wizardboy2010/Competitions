# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Total.csv')
data = data.drop(['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15'], 1)
tdata = data[data.iloc[:,-5] == 'Town'].drop(['City.Type'],1)
cdata = data[data.iloc[:,-5] != 'Town'].drop(['City.Type'],1)

cdata.to_csv('city.csv')
tdata.to_csv('town.csv')

data['sales_per_pop'] = data['Sale']/data['Population']

x =  data.groupby(pd.cut(data.sales_per_pop, np.percentile(data.sales_per_pop, [0, 25, 75, 90, 100]), include_lowest=True)).mean()

pd.cut(data.sales_per_pop, np.percentile(data.sales_per_pop, [0, 25, 75, 90, 100]), include_lowest=True)

p = np.percentile(data.sales_per_pop, [0, 25, 75, 90, 100])

data['group_percentile'] = pd.cut(data.sales_per_pop, np.percentile(data.sales_per_pop, [0, 25, 75, 90, 100]), include_lowest=True)

c = np.ones([429])
for i in range(len(data)):
    if data['group_percentile'][i]==p[0]:
        c[i] = 1
    if data['group_percentile'][i]==p[1]:
        c[i] = 2
    if data['group_percentile'][i]==p[2]:
        c[i] = 3
    if data['group_percentile'][i]==p[3]:
        c[i] = 4
    if data['group_percentile'][i]==p[4]:
        c[i] = 5

np.array(c)