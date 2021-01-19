from scipy import sparse as sp
from numpy import random,mat,ndarray

import tensorflow as tf
s = tf.Variable([[[1,3,2],[4,5,6],[7,8,9]],[[1,3,2],[4,5,6],[7,8,9]]], dtype=tf.float32)

mean, variance = tf.nn.moments(s, [0,1])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    mean, variance= sess.run([mean, variance])
    print('mean =',mean)
    print('variance =',variance)