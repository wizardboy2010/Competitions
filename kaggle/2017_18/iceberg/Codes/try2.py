# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.InteractiveSession()

train_df = pd.read_json('train.json')

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])

x = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y = np.array(train_df["is_iceberg"])

from sklearn.preprocessing import OneHotEncoder
oey = OneHotEncoder()
y = oey.fit_transform(y.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

X = tf.placeholder(tf.float32, [None, 75, 75, 2])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 2])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200 # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 2, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
#W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
#B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
#W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
#B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([25 * 25 * K, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [2]))

#stride = 1 ........output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 3, 3, 1], padding='SAME') + B1)
#stride = 2 ........output is 14x14
#Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
#stride = 2 ........output is 7x7
#Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)
#Pooling layer.....stride = 2 ........output is 8x8
#Y2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y1, shape=[-1, 25 * 25 * K])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

#initialize tf variables and session 
sess.run(tf.global_variables_initializer())

train_cost = []
test_cost = []
train_acc = []
test_acc = []

for i in range(100):  
   train_step.run(feed_dict={X: x_train, Y_: y_train, lr: 0.001, pkeep: 0.75})
   trc = cross_entropy.eval(feed_dict={X: x_train, Y_: y_train, pkeep: 0.75})
   tec = cross_entropy.eval(feed_dict={X: x_test, Y_: y_test, pkeep: 0.75})
   tra = accuracy.eval(feed_dict={X: x_train, Y_: y_train, pkeep: 0.75})
   tea = accuracy.eval(feed_dict={X: x_test, Y_: y_test, pkeep: 0.75})
   train_cost.append(trc)
   test_cost.append(tec)
   train_acc.append(tra)
   test_acc.append(tea)
   print("For epoch ", i+1, ",Train cost:", trc, ",test cost:", tec,",train acc:", tra,",test acc:", tea)
   
plt.plot(train_cost, 'blue')
plt.plot(test_cost, 'red')
plt.show()
plt.plot(train_acc, 'blue')
plt.plot(test_acc, 'red')
plt.show()
