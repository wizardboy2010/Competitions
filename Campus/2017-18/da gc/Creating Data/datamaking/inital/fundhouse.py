#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:19:50 2018

@author: user
"""

import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import tqdm as tqdm
import os
import pickle as pkl
from calendar import month_name as mn

data = pkl.load(open('fundtype/fundwise.pkl', 'rb'))
house = pkl.load(open('housenames.pkl', 'rb'))

def sep(temp, ids):
    t = []
    c = 0
    for i in ids:
        t.append(temp[c:i])
        c = i
    return t


def create(given, val, names, data, ids, n):
    temp = []
    for i in tqdm.tqdm(range(len(names))):
        c = 0
        for j in range(len(given)):
            if given[j] in names[i]:
                temp.append(val[j])
                c = 1
        if c == 0:
            temp.append(np.nan)
    f = sep(temp, ids)
    for i in range(12):
        data[i][n] = f[i]
    return data


tempdata = data

#tempdata = create(d['date'].values, d['CPI_urban'].values, scheme, tempdata, ids, 'Fund House')

for i in range(12):
    tempdata[i]['Fund House'] = house[i].values

pkl.dump(tempdata, open('fundtype/fundwise.pkl', 'wb'))

d = pd.read_csv('currency.csv')

temp = d['Date'].values

a = pd.DataFrame(columns = [0,1,2])
for i in range(len(temp)):
    a.loc[i] = [temp[i][:2], temp[i][3:5], temp[i][6:]]

for i in range(len(temp)):
    a['Date'][i] = a.iloc[i,0]+'-'+mn[int(a.iloc[i,1])][:3]+'-'+a.iloc[i,2]

d['Date'] = a['Date'].values

#schemes = []
dates = []
for i in range(12):
    dates.append(data[i]['Date'].values)
    #schemes.append(data[i]['Scheme Name'].values)


ids = []
c = 0
for i in dates:
    c += len(i)
    ids.append(c)

#stot = []
dtot = []
for i in range(len(dates)):
    #stot = np.append(stot, schemes[i])
    dtot = np.append(dtot, dates[i])

col = d.columns[1:]

for i in col:
    data = create(d['Date'].values, d[i].values, dtot, data, ids, i)
    
pkl.dump(data, open('fundtype/fundwise.pkl', 'wb'))
######################################################################

f = os.listdir('economic')

d = pd.read_csv('economic'+'/'+f[0])
d = d.drop(d.index[0])
col = d.iloc[0,:]
d = d.drop(d.index[0])

d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d.iloc[0,i]+'-'+str(d.iloc[1,i])