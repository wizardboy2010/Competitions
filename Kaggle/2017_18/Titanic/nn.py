#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:39:08 2017

@author: user
"""

from preprocessing import x,y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x.astype(float), y.astype(float), test_size = 0.2, random_state = 20)

import tensorflow as tf

x_ = tf.placeholder(tf.float64, name = "X")
y_ = tf.placeholder(tf.float64, name = "Y")

with tf.name_scope("layer1"):
    w1 = tf.Variable(tf.random_normal([4,3],dtype = tf.float64), name = "W")
    b1 = tf.Variable(tf.random_normal([3], dtype = tf.float64), name = "B")
    l1 = tf.nn.relu(tf.matmul(x_,w1)+b1, name = "l1")
    tf.summary.histogram("weights",w1)
    tf.summary.histogram("bias",b1)
with tf.name_scope("layer2"):
    w2 = tf.Variable(tf.random_normal([3,2],dtype = tf.float64), name  = "W")
    b2 = tf.Variable(tf.random_normal([2],dtype = tf.float64), name = "B")
    l2 = tf.nn.sigmoid(tf.matmul(l1,w2)+b2, name = "l2")
    tf.summary.histogram("weights",w1)
    tf.summary.histogram("bias",b1)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = l2, labels = y_))
    tf.summary.scalar('cost', cost)
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(l2,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('temp/summary/2', sess.graph)
    merge_sum = tf.summary.merge_all()
    init.run()
    for i in range(1000):
        sess.run(optimizer, feed_dict={x_: x_train, y_: y_train})
        s= sess.run(merge_sum, feed_dict={x_: x_test, y_: y_test})
        writer.add_summary(s,i)

print("After optimization : ")
training_cost = sess.run(cost, feed_dict={x_: x_train, y_: y_train})
print("Training cost= ", training_cost)
training_acc = sess.run(accuracy, feed_dict={x_: x_train, y_: y_train})
print("Training accuracy= ", training_acc)
test_acc = sess.run(accuracy,feed_dict={x_: x_test, y_:y_test})
print("test accuracy : ", test_acc)


