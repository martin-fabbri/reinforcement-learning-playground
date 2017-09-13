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
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
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


sess = tf.Session()
#with tf.Session() as sess:
sess.run(init)
print(sess)
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
        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()
        
        # get new state and reward from environment
        s1, r, done, _ = env.step(a[0])
        # obtain the q' values by feeding the new state throgh our network
        Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1: s1+1]})
        
        # obtain maxQ' and set our target value for chosen action
        maxQ1 = np.max(Q1)
        targetQ = all_q
        targetQ[0, a[0]] = r + y * maxQ1
        _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s1: s1+1], nextQ: targetQ})
        r_all += r
        s = s1
        if done:
            # reduce chance of random action as we train
            e = 1. / ((i / 50) + 10)
            break
    j_list.append(j)
    r_list.append(r_all)

print(f"Percent of successful episodes: {sum(r_list)/num_episodes}")

plt.plot(r_list)

plt.plot(j_list)

plt.show()
