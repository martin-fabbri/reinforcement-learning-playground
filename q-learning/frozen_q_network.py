import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# load the gym environment
env = gym.make("FrozenLake-v0")

# implement Q-Network approach
tf.reset_default_graph()

# feed-forward part of the network used to
# choose actions
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# bellow we obtain the loss by taking the sum of squares
# difference between the target and prediction Q values
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

updateModel = trainer.minimize(loss)

# training the network
init = tf.initialize_all_variables()

# set learning parameters
y = .99
e = 0.1
num_episodes = 2000

# create lists to contain total rewards and steps per episode
j_list = []
r_list = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # reset environment and get first new observation
        s = env.reset()
        r_all = 0
        d = False
        j = 0
        # the q-network
        while j < 99:
            j += 1
            # choose an action by greedily
            a, all_q = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s: s + 1]})
