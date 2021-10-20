# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:04:04 2021

Cite:
“Train Q-Learning Agent with Python - Reinforcement Learning Code Project.” Deeplizard, https://deeplizard.com/learn/video/HGeI30uATws. 

@author: jyanc
"""

import numpy as np
import gym
import random

env = gym.make("FrozenLake-v1")

action_size = env.action_space.n
state_size = env.observation_space.n

q_table = np.zeros((state_size, action_size))
print(q_table)
env.render()

num_episodes=10000
max_steps_per_episode=100

learning_rate=0.1           #alpha
discount_rate=.99           #gamma

exploration_rate=1          #epsilon
max_exploration_rate=1
min_exploration_rate=.01
exploration_decay_rate=.001

rewards_all_episodes=[]

#Q-learning algorithm
for episode in range(num_episodes):
    state=env.reset()
    
    done=False
    rewards_current_episode=0
    
    for step in range(max_steps_per_episode):
        #Exploration-exploitation trade-off
        exploration_rate_threshold=random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action=np.argmax(q_table[state,:])
        else:
            action=env.action_space.sample()
            
        new_state, reward, done, info=env.step(action)
        
        #update Q-table for Q(s,a)
        q_table[state,action]=q_table[state,action]*(1-learning_rate)+\
            learning_rate*(reward+discount_rate*np.max(q_table[new_state,:]))
            
        state=new_state
        rewards_current_episode += reward
            
        if done == True:
            break
            
        #exploration decay
    exploration_rate=min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
            
    rewards_all_episodes.append(rewards_current_episode)
        
#calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes=np.split(np.array(rewards_all_episodes),num_episodes/1000)
count=1000
print("*****Average reward per thousand episodes*****\n")
for r in rewards_per_thousand_episodes:
    print(count,": ",str(sum(r/1000)))
    count += 1000
    
    #print updated Q-table
print("\n\n*****Q-table*****\n")
print(q_table)
    
#code to watch the agent play frozen lake
for episode in range(3):
    state=env.reset()
    done=False
    print("*****Episode ", episode+1, "*****\n\n\n")
    #time.sleep(1)
    
    for step in range(max_steps_per_episode):
        #clear_output(wait=True)
        env.render()
        #time.sleep(0.3)
        
        action=np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        
        if done:
            #clear_output(wait=True)
            env.render()
            if reward==1:
                print("*****You reched the goal!*****")
               # time.sleep(3)
            else:
                print("*****You fell through a hole!*****")
                #time.sleep(3)
            #clear_output(wait=True)
            print("Number of steps", step)
            break
        
        state=new_state
        
env.close()

#alternative code for agent to play
env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps_per_episode):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()