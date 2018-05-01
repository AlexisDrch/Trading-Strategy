"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q_table = np.zeros((num_states, num_actions))
        if self.dyna > 0 :
            self.Tc = np.full((num_states, num_actions, num_states), fill_value = 0.00001)
            self.T = np.zeros((num_states, num_actions, num_states))
            self.R = np.zeros((num_states, num_actions))

    def author(self):
        return 'adurocher3' # replace tb34 with your Georgia Tech username.

    def getQ(self):
        print("rar", self.rar)
        print("raddr", self.radr)
        return self.Q_table

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if rand.uniform(0.0, 1.0) <= self.rar:
            # random action
            action = rand.randint(0, self.num_actions-1)
        else :
            # policy action
            action = np.argmax(self.Q_table[s, :])

        if self.verbose: print("s =", s,"a =",action)
        self.a = action
        self.rar = self.rar * self.radr
        
        return action

    def updateQTable(self, s, a, s_prime, r):
        self.Q_table[s, a] = (1 - self.alpha) * self.Q_table[s, a] + \
            self.alpha * (r + self.gamma * np.max(self.Q_table[s_prime]))
        
    def updateTc(self, s_prime):
        self.Tc[self.s, self.a, s_prime] += 1
        
    def updateT(self, s_prime):
        self.T[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] / np.sum(self.Tc[self.s, self.a])

    def updateR(self, r):
        self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha * r

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        # modeling the real world with T and R
        self.updateQTable(self.s, self.a, s_prime, r)

        if self.dyna >0 :
            self.updateTc(s_prime)
            self.updateT(s_prime)
            self.updateR(r)
            s_hallu = np.random.randint(0, self.num_states, self.dyna)
            a_hallu = np.random.randint(0, self.num_actions, self.dyna)

            for i in range(0, self.dyna):
                # hallucination
                s_hallu_i = s_hallu[i] 
                a_hallu_i = a_hallu[i]
                # inference from T model
                s_prime_hallu = np.argmax(self.T[s_hallu_i, a_hallu_i])
                r_hallu = self.R[s_hallu_i, a_hallu_i]
                self.updateQTable(s_hallu_i, a_hallu_i, s_prime_hallu, r_hallu)
            
        return self.querysetstate(s_prime)

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
    
#author adurocher3 Alexis DUROCHER