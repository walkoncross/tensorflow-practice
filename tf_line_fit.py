import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 + 0.01*np.random.rand(100).astype(np.float32)

plt.plot(x_data, y_data, 'b.')
plt.title('real line')
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.show()

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print("Iter {}: W={}, b={}, loss={}".format(step, sess.run(W), sess.run(b), sess.run(loss)))

plt.figure()
plt.plot(x_data, y_data, 'b.')
plt.hold()
plt.plot(x_data, sess.run(y), 'r')
plt.title('fitted line')
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.show()

# Learns best fit is W: [0.1], b: [0.3]