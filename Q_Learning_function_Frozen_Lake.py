# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:25:18 2021

@author: jyanc
"""

import numpy as np
import gym
import random

def Q_Learning(env, total_episodes, learning_rate, max_steps, gamma):
    action_size=env.action_space.n
    state_size=env.observation_space.n
    
    qtable=np.zeros((state_size,action_size))
    print(qtable)
    env.render()
    
    #exploration parameters
    epsilon=1.0                     #exploration rate
    max_epsilon=1.0                 #exploration probability at start
    min_epsilon=0.01                #min exploration probability
    decay_rate=.001                 #Exponential decay rate for exploration prob
    
    #list rewards
    rewards=[]
    
    for episode in range(total_episodes):
        state=env.reset()
        step=0
        done=False
        total_rewards=0
        
        for step in range(max_steps):
            exp_tradeoff=random.uniform(0,1)
            if exp_tradeoff>epsilon:
                action=np.argmax(qtable[state,:])
            else:
                action=env.action_space.sample()
            
            new_state, reward, done, info=env.step(action)
            
            qtable[state,action]=(1-learning_rate)*qtable[state,action]+learning_rate*(reward+gamma*np.max(qtable[new_state,:]))
            
            total_rewards+=reward
            state=new_state
            
            if done==True:
                break
        epsilon=min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
        rewards.append(total_rewards)
        
    #print("Score over time: ",+ str(sum(rewards)/total_episodes))
    print(qtable)

