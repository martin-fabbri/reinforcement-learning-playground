"""
The four-armed bandit - Policy Gradient¶
Policy gradient based agent that solves a two-armed problem.
"""

import tensorflow as tf
import numpy as np

# list of bandits
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)


def pull_bandit(bandit):
    result = np.random.rand(1)
    if result > bandit:
        return 1
    else:
        return -1

# the agent
tf.reset_default_graph()

# establish feed-forward
weights = tf.Variable(tf.ones([num_bandits]))
choose_action = tf.argmax(weights, 0)

# training
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

total_episodes = 1000
total_reward = np.zeros(num_bandits)

e = 0.1

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        rand = np.random.rand(1)
        if rand < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(choose_action)

        reward = pull_bandit(bandits[action])

        _, resp, ww = sess.run([update, responsible_weight, weights],
                               feed_dict={reward_holder: [reward], action_holder: [action]})

        total_reward[action] += reward
        if i % 50 == 0:
            print(f"running reward for bandits: {total_reward}")

        i += 1


print(f"The agent thinks bandit {np.argmax(ww) + 1} is the most promising.")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right.")
else:
    print("...and it was wrong.")