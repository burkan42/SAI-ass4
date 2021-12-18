#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self,env,gamma = 1.0, theta=0.001):
        pass
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''
        
        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)
        
        while True:
            biggest_value = 0
            for s in env.states:
                
                V_old = V_s[s]
                V_new = 0
                
                for a in env.actions:
                    vector_s, reward_s = env.transition_function(s,a)
                    
                    v = reward_s + (gamma * V_s[vector_s])
                    if v > V_new:
                        V_new = v
                
                V_s[s] = V_new
                biggest_value = max(biggest_value, np.absolute(V_old -V_new))
            if biggest_value < theta:
                break
        print(V_s)
        self.V_s = V_s
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        while True:
            biggest_value = 0
            for s in env.states:
                
                Q_old = Q_sa[s]
                Q_new = 0
                
                for a in env.actions:
                    next_state, reward_s = env.transition_function(s,a)
                    best_next_action = np.argmax(Q_sa[next_state])
                    v = reward_s + (gamma * Q_sa[next_state][best_next_action]) - Q_sa[s][a]
                    if v > Q_new:
                        Q_new = v
                
                Q_sa[s][a] = Q_new
                biggest_value = max(biggest_value, np.absolute(Q_old - Q_new))
            if biggest_value < theta:
                break

        self.Q_sa = Q_sa
        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:           

                V_new = 0
                for a in available_actions:
                    next_state, reward_s = env.transition_function(current_state, a)
                    
                    v = reward_s + self.V_s[next_state]
                    if v > V_new:
                        V_new = v
                        best_action = a
                             
                
                greedy_action = best_action

            elif table == 'Q' and self.Q_sa is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.Q_sa here !!!
                
                print("You still need to implement greedy action selection from the state-action value table self.Q_sa!")
                greedy_action = None # replace this!
                
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

