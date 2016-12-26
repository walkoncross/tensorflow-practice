# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 08:14:21 2016

@author: zhaoy
"""
import tensorflow as tf

# Create some variables.
v1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="v1")
v2 = tf.Variable(tf.zeros([1]), name="v2")

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  # ...
  # Save the variables to disk.
  save_path = saver.save(sess, "./model.ckpt")
  print("Model saved in file: %s" % save_path)
#Restoring Variables

#The same Saver object is used to restore variables. Note that when you restore variables from a file you do not have to initialize them beforehand.

# Create some variables.
v1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="v1")
v2 = tf.Variable(tf.zeros([1]), name="v2")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./model.ckpt")
  print("Model restored.")
  # Do some work with the model
