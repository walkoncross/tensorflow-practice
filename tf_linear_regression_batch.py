# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 06:29:04 2017

@author: zhaoy
"""


import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

# Define input data
x_data = np.arange(100, step=.1)
y_data = x_data + 20*np.sin(x_data/10)

# Plot input data 
#plt.scatter(x_data, y_data)
plt.plot(x_data, y_data, 'b-')

# Define data size and batch size
n_samples = 1000
batch_size = 100
n_iter = 100

# Tensorflow is finicky about shapes, so resize
x_data = np.reshape(x_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

tf.reset_default_graph()

# Define placeholders for input
#x = tf.placeholder(tf.float32, shape=(n_samples, 1))
#y = tf.placeholder(tf.float32, shape=(n_samples, 1))
#
x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))


# Define variables to be learned
with tf.variable_scope('linear-regression', reuse=None):
    w = tf.get_variable('weights', (1, 1),
                        initializer=tf.random_normal_initializer())
    
    b = tf.get_variable('bias', (1, ), 
                        initializer=tf.constant_initializer(0.0))
    
    y_pred = tf.matmul(x, w) + b
    
    #loss = tf.reduce_sum((y-y_pred)**2/n_samples)
    loss = tf.reduce_mean(tf.square(y - y_pred))
   
## Sample code to run one step of gradient descent
#opt = tf.train.AdamOptimizer()
#
#opt_operation = opt.minimize(loss)
#
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    sess.run([opt_operation], feed_dict = {x:x_data, y:y_data})
#    print('===>type(w.eval()): ' + str(type(w.eval())))
#    print('===>type(b.eval()): ' + str(type(b.eval())))
#    print('===>w.eval(): ' + str(w.eval()))
#    print('===>b.eval(): ' + str(b.eval()))
#    
#    y_pred_val = x_data * (w.eval()) + b.eval()
#    #plt.figure()
#    plt.hold(True)
#    plt.plot(x_data, y_pred_val, 'r-')
    

# Sample code to run one step of gradient descent
# Define optimizer operation
opt_operation = tf.train.AdamOptimizer().minimize(loss)
#opt_operation = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
    # Initialize variables in graph
    #sess.run(tf.initialize_all_variables())
    init = tf.global_variables_initializer()
    sess.run(init)
    # Gradient descent loop for 500 steps
    for i in range(n_iter):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        x_batch, y_batch = x_data[indices], y_data[indices]
        # Do gradient descent step    
        _, loss_val = sess.run([opt_operation, loss], feed_dict = {x:x_batch, y:y_batch}) 
        
        if i%10==0:
            print('===>Iter {}: train_loss = {}'.format(i, loss.eval(feed_dict={x:x_batch, y:y_batch})))
    
    print('===>type(w.eval()): ' + str(type(w.eval())))
    print('===>type(b.eval()): ' + str(type(b.eval())))
    print('===>w.eval(): ' + str(w.eval()))
    print('===>b.eval(): ' + str(b.eval()))
    
    y_pred_val = x_data * (w.eval()) + b.eval()
    #plt.figure()
    plt.hold(True)
    plt.plot(x_data, y_pred_val, 'r-')
