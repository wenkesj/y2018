#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test that runs the MNIST dataset against the dense + notdense final classifier.

Set `use_dense = True` to compare to dense + dense final classifier.
"""
from __future__ import print_function
from __future__ import division

from y2018.notdense import notdense
from y2018.utils import generator

from observations import mnist

import tensorflow as tf

import time

dense = tf.layers.dense

epochs = 10
batch_size = 512
learning_rate = 1e-3
use_dense = False

(x_train, y_train), (x_test, y_test) = mnist('data')
train_size, test_size = len(x_train), len(x_test)
train_set, test_set = (
  generator([x_train, y_train], batch_size),
  generator([x_test, y_test], batch_size),
)

inputs = tf.placeholder(tf.uint8, (None, 784), name='inputs')
labels = tf.placeholder(tf.int32, (None,), name='labels')

y = dense(
    2 * (tf.cast(inputs, tf.int32) / 255) - 1, units=256,
    activation=tf.nn.tanh,
    use_bias=False,
    kernel_initializer=tf.initializers.variance_scaling())

if use_dense:
  logits = dense(
      y, units=10,
      use_bias=False)
else:
  logits = notdense(
      y, units=10,
      use_bias=False,
      fixed_q=True,
      fixed_alpha=True,
      alpha=10.)

pred_op = tf.argmax(logits, axis=-1, output_type=tf.int32)
acc_op = tf.reduce_mean(
    tf.cast(tf.equal(pred_op, labels), tf.float32), axis=-1)
loss_op = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='loss'),
    axis=-1)
train_op = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(loss_op)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(epochs):
    elapsed = 0
    for it, (x, y) in enumerate(train_set):
      if it > (train_size // batch_size):
        break
      start = time.time()
      loss, _ = sess.run((loss_op, train_op), feed_dict={
        inputs: x,
        labels: y,
      })
      elapsed += (time.time() - start)
    print('avg time {}'.format(elapsed / it))
    accy = 0
    for it, (x, y) in enumerate(test_set):
      if it > (test_size //  batch_size):
        break
      accy += sess.run(acc_op, feed_dict={
        inputs: x,
        labels: y
      })
    print('test accy. at epoch {} = {:.2f}%'.format(epoch, 100. * (accy / it)))
