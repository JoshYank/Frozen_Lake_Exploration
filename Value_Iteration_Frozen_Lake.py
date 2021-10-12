# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:36:16 2021

@author: jyanc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:05:20 2021
Credit:
    Segu, Sugun. “Value Iteration to Solve Openai Gym's FrozenLake.” Medium, Towards Data Science, 14 June 2020, https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d. 

@author: jyanc
"""

import numpy as np
import tools
import matplotlib.pyplot as plt
import gym
env=gym.make("FrozenLake8x8-v1")
#8x8 frozen lake

env.env.nA

#Out[23]: 4

env.action_space
#Out[24]: Discrete(4)

env.env.nS
#Out[25]: 64

env.env.P[0][1]
"""
Out[26]: 
[(0.3333333333333333, 0, 0.0, False),
 (0.3333333333333333, 8, 0.0, False),
 (0.3333333333333333, 1, 0.0, False)]
"""
def argmax(env,V,pi,s,gamma):
    e=np.zeros(env.env.nA)
    for a in range(env.env.nA):        #iterate for every action possible
        q=0
        P=np.array(env.env.P[s][a])
        (x,y)=np.shape(P)              #for Bellman eq

        for i in range(x):             #iterate for every possible state
            s_=int(P[i][1])            #s'-sprime-possible successive state
            p=P[i][0]                  #Transition Probability P(s'|s,a)
            r=P[i][2]                  #reward

            q += p*(r+gamma*V[s_])     #calc action_ value q(s|a)
            e[a]=q

    m=np.argmax(e)                     #take index which has max value
    pi[s][m]=1                         #update pi(a|s)

    return pi
    

def Bellman_optimality_update(env,V,s,gamma): ## update the stae_value V[s] by taking
    pi=np.zeros((env.env.nS,env.env.nA))       # action which maximizes current value
    e=np.zeros((env.env.nA))
                                                # STEP1: Find
    for a in range(env.env.nA):
        q=0                                     # iterate for all possible action
        P=np.array(env.env.P[s][a])
        (x,y)=np.shape(P)

        for i in range(x):
            s_=int(P[i][1])
            p=P[i][0]
            r=P[i][2]
            q += p*(r+gamma*V[s_])
        e[a]=q

    m=np.argmax(e)
    pi[s][m]=1


    u=0
    P=np.array(env.env.P[s][m])
    (x,y)=np.shape(P)

    for i in range(x):
        s_=int(P[i][1])
        p=P[i][0]
        r=P[i][2]

        u += p*(r+gamma*V[s_])
    V[s]=u
    return V[s]
    

def value_iteration(env,gamma,theta):
    V=np.zeros(env.env.nS)                                # initialize v(0) to arbitory value, my case "zeros"
    while True:
        delta=0
        for s in range(env.env.nS):                        # iterate for all states
            v=V[s]
            Bellman_optimality_update(env,V,s,gamma)       # update state_value with bellman_optimality_update
            delta=max(delta,abs(v-V[s]))                    # assign the change in value per iteration to delta
        if delta<theta: 
            break                                            # if change gets to negligible 
                                                            # --> converged to optimal value
    pi=np.zeros((env.env.nS,env.env.nA))
    for s in range(env.env.nS):
        pi=argmax(env,V,pi,s,gamma)                        # extract optimal policy using action value

    return V, pi                                          # optimal value funtion, optimal policy
    

gamma=.99
theta=.0001
V,pi=value_iteration(env,gamma,theta)

env.render()

tools.plot(V,pi)

#where does action come from?

a= np.reshape(action,(4,4))
print(a)                          # discrete action to take in given state


e=0
for i_episode in range(100):
    c = env.reset()
    for t in range(10000):
        c, reward, done, info = env.step(action[c])
        if done:
            if reward == 1:
                e +=1
            break
print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e+1))
env.close()