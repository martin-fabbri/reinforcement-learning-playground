## Introduction

### Learning Goals

- Understand the Reinforcement Learning problem and how it differs from Supervised Learning.

### Summary

TODO

### Characteristics of Reinforcement Learning

What makes reinforcement learning different from other machine learning paradigms?

- There is no supervision, only a reward signal. No feedback about what is the best or correct course of actions. But rather trial and error experiments; reward signals that helps to interact and understand the environment.

- Feedback is delayed not instantaneous. It might be delayed for many steps.
- Time really matters. We are dealing with sequential processes.
- Agent's actions affect the subsequent data it receives (Active learning). 


> **Definition (Reward Hypothesis)**
>
> All goals can be described by the maximization of expected cumulative rewards.

### Reward

- A reward is a R(t) scalar feedback signal.
- Indicates how well the agent is doing at time t.
- The agent's goal is to maximize the cumulative reward.


### Lectures and Readings

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/the-book-2nd.html) - Chapter 1: Introduction
- [David Silver's RL Course Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Introduction to Reinforcement Learning ([slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf))