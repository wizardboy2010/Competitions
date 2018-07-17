# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 13:27:43 2017

@author: intel
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf

data = pd.read_csv('Training_Dataset.csv')

#data_1 = data.drop('mvar40', axis=1)
#data_1 = data_1.drop('mvar42', axis=1)
#data_1 = data_1.drop('mvar44', axis=1)
data_1 = data.drop('mvar46', axis=1)
data_1 = data_1.drop('mvar47', axis=1)
data_1 = data_1.drop('mvar48', axis=1)
data_1 = data_1.drop('cm_key', axis=1)
data_1 = data_1.drop('mvar1', axis=1)

x = data_1.iloc[:,:-3].values
y = data_1.iloc[:,-3:].values

y = np.append(y,np.ones((40000,1),dtype = 'int'), axis=1)
for i in range(40000):
    if y[i,0]==1 or y[i,1]==1 or y[i,2]==1:
        y[i,3]=0

#pre_processing
x = np.delete(x,1,1)
x = np.delete(x,9,1)
x= x.astype(np.float64)
from sklearn.preprocessing import Imputer, StandardScaler
imputer = Imputer(missing_values = 0,strategy = 'mean',axis = 0)
imputer = imputer.fit(x[:,6].reshape(-1,1))
x[:,6] = np.reshape(imputer.transform(x[:,6].reshape(-1,1)),(40000,))
#imputer = imputer.fit(x[:,1].reshape(-1,1))
#x[:,1] = np.reshape(imputer.transform(x[:,1].reshape(-1,1)),(40000,))

sc = StandardScaler()
x[:,:] = sc.fit_transform(x[:,:])
#x[:,3:9] = sc.fit_transform(x[:,3:9])
#x[:,10:-9] = sc.fit_transform(x[:,10:-9])
'''x=np.insert(x,2,np.zeros(40000),axis = 1)
x=np.insert(x,9,np.zeros(40000),axis = 1)
a=np.mean(x[:,1])
b=np.mean(x[:,8])
for i in range(40000):
    if x[i,1]==0:
        x[i,2]=1
    if x[i,8]==0:
        x[i,9]=1'''
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state = 10)

'''#model1
def relu(x):
    return (x+np.absolute(x))/2
def softplus(x):
    y = np.log(1+np.exp(x))
    return y
def softmax(x):
    y = np.exp(x)/sum(np.exp(x))
    return y
    
h1 = 21
a= x_train.shape[1]
b= y_train.shape[1]
w1 = np.random.rand(a,h1)
b1 = np.random.rand(h1)
w2 = np.random.rand(h1,b)
b2 = np.random.rand(b)
m = list([w1,b1,w2,b2])
def create(m,x):
    p = x.shape[0]
    l1_logit = np.matmul(x,m[0])
    l1 = relu(l1_logit + m[1])
    l2_logit = np.matmul(l1,m[2])
    output = np.zeros([p,4])
    for i in range(p):
        output[i, :] = softmax(l2_logit[i,:] + m[3])
    l = list([l1,output])
    return l
def cost(y,out):
    #error = np.square(y-out)
    #cost = np.mean(error)
    cross_entropy= -np.sum(y*np.log(out))
    cost = np.mean(cross_entropy)
    return cost
def derv(a):
    p=np.sign(a)
    return (p+np.absolute(p))/2
def optimize2(x,m,l,y,lr):
    r1 = np.zeros([21])
    #p = x.shape[0]
    r2 = (l[1]-y)
    dw2 = np.matmul((l[0].T),r2)
    r1 += np.sum(np.multiply(np.matmul(((derv(l[0])).T),r2),m[2]),axis=1)
    dw1 = np.zeros((39,21))           
    for j in range(21):
        for k in range(39):
            dw1[k,j] = r1[j]*np.mean(x[:,k])
    m[2] -= lr*dw2
    m[3] -= lr*np.mean(r2,axis = 0)
    m[0] -= lr*dw1
    m[1] -= lr*r1
    return m
class nn:
    def init(n,start,end):
        w = []
        w.append(np.random.rand(start+1,n[0]))
        for i in range(np.shape(n)[0]-1):
            w.append(np.random.rand(n[i]+1,n[i+1]))
        w.append(np.random.rand(n[-1]+1,end))
        return w
    def relu(x):
        return (x+np.absolute(x))/2
    def forprop(x,w):
        n = np.ones(len(w)-1)
        l = []
        for i in range(len(n)):
            n[i] = np.shape(w[i])[1]
        l.append(nn.relu(np.matmul(np.insert(x,0,1,axis=1),w[0])))
        for i in range(len(n)-1):
            l.append(nn.relu(np.matmul(np.insert(l[i],0,1,axis=1),w[i+1])))
        l.append(nn.softmax(np.matmul(np.insert(l[len(n)-1],0,1,axis=1),w[len(n)])))
        return l
    def softmax(x):
        return np.exp(x)/sum(np.exp(x))
    def grad_disc(x,y,w,l,lr):
        p = x.shape[0]
        n = np.ones(len(w)-1)
        for i in range(len(n)):
            n[i] = np.shape(w[i])[1]
        d = []
        D = []
        d.append(y-l[-1])
        for i in range(len(n)):
            d.append(np.multiply(np.matmul(d[-(i+1)],np.delete(w[-(i+1)],0,0).T),nn.relu(l[-(l+2)])))
        D.append(np.matmul(x.T,d[-1])/p)
        for i in range(len(n)):
            D.append(np.matmul(l[i].T,d[-(i+2)])/p)
        w -= lr*D
        return w'''
def accuracy(y_pred, y_true):
    a = y_pred.shape[0]
    pred = np.zeros([a])
    true = np.zeros([a])
    for i in range(a):
        pred[i] = np.argmax(y_pred[i,:])
        true[i] = np.argmax(y_true[i,:])
    accuracy = accuracy_score(true,pred)
    return accuracy
'''n = [21,21]
w = nn.init(n,39,4)
layers = nn.forprop(x_train,w)
n_epoch = 15
accu_test = np.ones([n_epoch]) 
accu_train = np.ones([n_epoch]) 
c_test = np.ones([n_epoch])
c_train = np.ones([n_epoch])
for i in range(n_epoch):
    #m = optimize2(x_train,m,layers,y_train,0.001)
    #layers = create(m,x_train)
    #layers_test = create(m,x_test)
    w = nn.grad_disc(x_train,y_train,w,layers,0.001)
    layers = nn.forprop(x_train,w)
    layers_test = nn.forprop(x_test,w)
    accu_train[i] = accuracy(layers[1],y_train)
    accu_test[i] = accuracy(layers_test[1],y_test)
    c_train[i] = cost(y_train, layers[1])
    c_test[i] = cost(y_test, layers_test[1])
    print('for',i+1,'train accuracy:',accuracy(layers[1],y_train),'test accuracy:',accuracy(layers_test[1],y_test))
'''

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-3,
                     hidden_layer_sizes=(21, 21), random_state=1)
clf.fit(x_train,y_train)
print('train accuracy is ',accuracy(clf.predict(x_train),y_train),'test accuracy is',accuracy(clf.predict(x_test),y_test))



out_data = pd.read_csv('Leaderboard_Dataset.csv')
#data_1 = out_data.drop('mvar40', axis=1)
#data_1 = data_1.drop('mvar42', axis=1)
#data_1 = data_1.drop('mvar44', axis=1)
data_2 = out_data.drop('cm_key', axis=1)
data_2 = data_2.drop('mvar1', axis=1)

x = data_2.iloc[:,:].values

#pre_processing
x = np.delete(x,1,1)
x = np.delete(x,9,1)
x= x.astype(np.float64)
from sklearn.preprocessing import Imputer, StandardScaler
imputer = Imputer(missing_values = 0,strategy = 'mean',axis = 0)
imputer = imputer.fit(x[:,6].reshape(-1,1))
x[:,6] = np.reshape(imputer.transform(x[:,6].reshape(-1,1)),(10000,))

sc = StandardScaler()
#x[:,:7] = sc.fit_transform(x[:,:7])
x[:,:] = sc.fit_transform(x[:,:])

out = np.ones((10000,3))
out[:,0] = out_data.iloc[:,0].values
r=clf.predict_proba(x)
for i in range(10000):
    out[i,1] = np.argmax(r[i,:])
    p = int(out[i,1])
    out[i,2] = r[i,p]
a=[]
for i in range(10000):
    if out[i,1] ==3.000:
        out[i,:] = np.zeros(3)
for i in range(10000):
    if out[i,0] ==0:
        a.append(i)
fout=np.delete(out,a,0)
fout = sorted(fout, key=lambda x:x[2])

for i in range(282):
    if fout[i,1]==0:
        fout[i,1]='Supp'
    if fout[i,1]==1:
        fout[i,1]='Elite'
    if fout[i,1]==2:
        fout[i,1]='Credit'
        
import xlsxwriter

workbook = xlsxwriter.Workbook('Phoenix_kgp_IITKharagpur_3.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(fout.T):
    worksheet.write_column(row, col, data)
workbook.close()


from fancyimpute import KNN