# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:33:43 2017

@author: intel
"""
import neuralnet
from neuralnet import *
import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('Training_Dataset.csv')

data_1 = data.drop('mvar40', axis=1)
data_1 = data_1.drop('mvar42', axis=1)
data_1 = data_1.drop('mvar44', axis=1)
data_1 = data_1.drop('mvar46', axis=1)
data_1 = data_1.drop('mvar47', axis=1)
data_1 = data_1.drop('mvar48', axis=1)
#data_1 = data_1.drop('cm_key', axis=1)
#data_1 = data_1.drop('mvar1', axis=1)

x = data_1.iloc[:,2:-3].values
y = data_1.iloc[:,-3:].values
y1 = np.zeros(x.shape[0])

for i in range(40000):
    if y[i,0]==1 or y[i,1]==1 or y[i,2]==1:
        y1[i]=1

x = np.delete(x,1,1)
from sklearn.preprocessing import Imputer, StandardScaler
imputer = Imputer(missing_values = 0,strategy = 'mean',axis = 0)
imputer = imputer.fit(x[:,6].reshape(-1,1))
x[:,6] = np.reshape(imputer.transform(x[:,6].reshape(-1,1)),(40000,))

x = np.delete(x,9,1)

sc = StandardScaler()
x[:,:7] = sc.fit_transform(x[:,:7])
x[:,8:-3] = sc.fit_transform(x[:,8:-3])
x= x.astype(np.float64)
from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y1, test_size = 0.2,random_state = 10)
'''
#forprop
x_ = tf.placeholder(tf.float64)
y_ = tf.placeholder(tf.float64)

w1 = tf.Variable(tf.random_normal([39,21], dtype = tf.float64))
b1 = tf.Variable(tf.random_normal([21],dtype = tf.float64))
#w2 = tf.Variable(tf.random_normal([21,21],dtype = tf.float64))
#b2 = tf.Variable(tf.random_normal([21],dtype = tf.float64))
w2 = tf.Variable(tf.random_normal([21,1],dtype = tf.float64))
b2 = tf.Variable(tf.random_normal([1],dtype = tf.float64))

l1 = tf.nn.relu(tf.add(tf.matmul(x_,w1),b1))
#l2 = tf.nn.relu(tf.add(tf.matmul(l1,w2),b2))
l_out = tf.nn.softmax(tf.add(tf.matmul(l1,w2),b2))

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = l_out, logits = y_))
cost = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(l_out,1e-10,1.0)))
#cost = tf.reduce_sum(tf.pow(y_ - l_out, 2))/(2*n_samples)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
correct_prediction = tf.equal(tf.argmax(l_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)
for i in range(10):
    sess.run(optimizer, feed_dict={x_: x_train, y_: y_train})
    print('i')
print("After optimization : ")
training_cost = sess.run(cost, feed_dict={x_: x_train, y_: y_train})
print("Training cost= ", training_cost)
training_acc = sess.run(accuracy, feed_dict={x_: x_train, y_: y_train})
print("Training accuracy= ", training_acc)
test_acc = sess.run(accuracy,feed_dict={x_: x_test, y_:y_test})
print("test accuracy : ", test_acc)

#print("W1= ", sess.run(w1), "b1= ", sess.run(b1))
#print("W3= ", sess.run(w3), "b3= ", sess.run(b3))

out_train = sess.run(l_out, feed_dict={x_:x_train})        '''

#decision tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
first = dt.fit(x,y1)
yfirst = dt.predict(x)
'''#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
first = rf.fit(x_train,y_train)
yfirst = rf.predict(x)
#boosting
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
first = ada.fit(x_train,y_train)
yfirst = ada.predict(x)
'''
x2 =[]
key = []
for i in range(len(yfirst)):
    if yfirst[i] ==0:
        x2.append(i)
    else:
        key.append(i)    
x2in = np.delete(x,x2,0)
y2in = np.delete(y,x2,0)
cm_key =[]
for i in range(len(key)):
    key[i] = data_1.iloc[key[i],0]
x_train, x_test, y_train, y_test = train_test_split(x2in,y2in, test_size = 0.3,random_state = 10)

'''#boosting
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
second = ada.fit(x_train,y_train)
ysecond = ada.predict(x)
'''
#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
second = rf.fit(x_train,y_train)
ysecond = rf.predict(x_test)

'''#neural net
x_ = tf.placeholder(tf.float64)
y_ = tf.placeholder(tf.float64)

w1 = tf.Variable(tf.random_normal([39,18], dtype = tf.float64))
b1 = tf.Variable(tf.random_normal([18],dtype = tf.float64))
#w2 = tf.Variable(tf.random_normal([21,21],dtype = tf.float64))
#b2 = tf.Variable(tf.random_normal([21],dtype = tf.float64))
w3 = tf.Variable(tf.random_normal([18,3],dtype = tf.float64))
b3 = tf.Variable(tf.random_normal([3],dtype = tf.float64))

l1 = tf.nn.relu(tf.add(tf.matmul(x_,w1),b1))
#l2 = tf.nn.relu(tf.add(tf.matmul(l1,w2),b2))
l_out = tf.nn.softmax(tf.add(tf.matmul(l1,w3),b3))

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = l_out, logits = y_))
cost = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(l_out,1e-10,1.0)))
#cost = tf.reduce_sum(tf.pow(y_ - l_out, 2))/(2*n_samples)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(l_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)
for i in range(3000):
    sess.run(optimizer, feed_dict={x_: x_train, y_: y_train})

print("After optimization : ")
training_cost = sess.run(cost, feed_dict={x_: x_train, y_: y_train})
print("Training cost= ", training_cost)
training_acc = sess.run(accuracy, feed_dict={x_: x_train, y_: y_train})
print("Training accuracy= ", training_acc)
test_acc = sess.run(accuracy,feed_dict={x_: x_test, y_:y_test})
print("test accuracy : ", test_acc)

out_train = sess.run(l_out, feed_dict={x_:x_test}) '''

#boosting
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(algorithm="SAMME")
third = ada.fit(x_test,ysecond)
#ythird = ada.predict(x2in)

out_data = pd.read_csv('Leaderboard_Dataset.csv')
data_2 = out_data.drop('mvar40', axis=1)
data_2 = data_2.drop('mvar42', axis=1)
data_2 = data_2.drop('mvar44', axis=1)
data_2 = data_2.drop('cm_key', axis=1)
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
x[:,:] = sc.fit_transform(x[:,:])
#x[:,8:-3] = sc.fit_transform(x[:,8:-3])

'''yfinal1 = dt.predict(x)
x2 =[]
key = []
for i in range(len(yfinal1)):
    if yfinal1[i] ==0:
        x2.append(i)
    else:
        key.append(i)    
x2in = np.delete(x,x2,0)
cm_key =[]
for i in range(len(key)):
    key[i] = out_data.iloc[key[i],0]'''

#y_final2 = rf.predict(x2in)
y_final3 = ada.predict(x)
#y_final = sess.run(l_out, feed_dict={x_:x2in})

out = np.ones((len(key),3))
out[:,0] = key
for i in range(len(key)):
    out[i,1] = np.argmax(y_final[i,:])
    p = int(out[i,1])
    out[i,2] = y_final[i,p]

'''import xlsxwriter

workbook = xlsxwriter.Workbook('Phoenix_kgp_IITKharagpur_9.csv')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(out.T):
    worksheet.write_column(row, col, data)
workbook.close()'''

np.savetxt("Phoenix_kgp_IITKharagpur_11.csv", out, delimiter=",", fmt='%s')