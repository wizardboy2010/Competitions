#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 23:58:18 2017

@author: user
"""

import tensorflow as tf

a = tf.constant(7, name='test_variable')
tf.summary.scalar('variable', a)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/tmp/summary', graph=sess.graph)
    X = tf.global_variables_initializer()
    sess.run(X)
    summary = sess.run(summary_op)
    summary_writer.add_summary(summary)
