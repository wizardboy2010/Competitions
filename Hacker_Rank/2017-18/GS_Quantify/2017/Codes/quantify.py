# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:34:42 2017

@author: preetish
"""

#quantify

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import r2_score


train=pd.read_csv('gcTrianingSet.csv')
test=pd.read_csv('gcPredictionFile.csv')

train['cpufreq']=np.sqrt(np.log(1/train['cpuTimeTaken']))
train['subr']=train['initialUsedMemory']-train['initialFreeMemory']
train['tot']=train['initialUsedMemory']+train['initialFreeMemory']
y1=train['gcRun'].astype(int)
y2=train['finalFreeMemory']
y3=train['finalUsedMemory']
X=train.drop(['gcRun','finalFreeMemory','finalUsedMemory','cpuTimeTaken'],axis=1)

temp=pd.get_dummies(X['query token'])#for token vectors(train)

temp2=pd.get_dummies(test['query token'])

from sklearn.decomposition import PCA

pca=PCA(n_components=22)
temp=pca.fit_transform(temp)
testtemp=pca.transform(temp)

X=X.drop(['query token','userTime','sysTime','realTime','gcInitialMemory','gcFinalMemory','gcTotalMemory'],axis=1)

X=np.array(X)
X=np.concatenate((X,temp),axis=1)

test=test.drop('query token',axis=1)

xtest=np.array(test)
xtest=np.concatenate((xtest,temp2),axis=1)

trainx,cvx,trainy1,cvy1=train_test_split(X,y2,test_size=0.33,random_state=42)

model1=SVR(C=0.25,gamma=0.4)

model1.fit(trainx,trainy1)
print(model1.score(trainx,trainy1))
m1=model1.predict(cvx)
t1=model1.predict(trainx)

model2=GradientBoostingRegressor(n_estimators=2000,learning_rate=0.005,max_depth=2,min_samples_split=35)
model2.fit(trainx,trainy1)
print(model2.score(trainx,trainy1))
t2=model2.predict(trainx)
m2=model2.predict(cvx)

m=np.sqrt(m1*m2)

test['cpufreq']=np.sqrt(np.log(1/test['cpuTimeTaken']))
test['subr']=test['initialUsedMemory']-test['initialFreeMemory']
test['tot']=test['initialUsedMemory']+test['initialFreeMemory']
test = np.array(test)

for i in range(1625):
    test(['finalFreeMemory']) = np.sqrt(model1.predict(test[i]))
























