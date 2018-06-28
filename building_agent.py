# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 20:11:01 2018
@author: lupus
"""

"""define winning function"""
def win(matrix):
    
    won = False
    
    if \
    abs(matrix[0] + matrix[1] + matrix[2]) == 3 or \
    abs(matrix[3] + matrix[4] + matrix[5]) == 3 or \
    abs(matrix[6] + matrix[7] + matrix[8]) == 3 or \
    abs(matrix[0] + matrix[3] + matrix[6]) == 3 or \
    abs(matrix[1] + matrix[4] + matrix[7]) == 3 or \
    abs(matrix[2] + matrix[5] + matrix[8]) == 3 or \
    abs(matrix[0] + matrix[4] + matrix[8]) == 3 or \
    abs(matrix[2] + matrix[4] + matrix[6]) == 3:
        
        won = True
    
    return won
"""define winning function"""


import random
import numpy as np
import matplotlib.pyplot as plt

max_episode = 20000
episode = 1
seen_states = {}
winning_list = []

while episode <= max_episode:
    matrix = [0]*9
    episode_states = []
    
    while 0 in matrix:
        #black move
        possible_action_list = [i for i, v in enumerate(matrix) if v == 0]
        possible_action_expected_reward = [-max_episode]*9
                    
        for possible_action in possible_action_list:
            possible_state = matrix.copy()
            possible_state[possible_action] = 1
            
            if str(possible_state) not in seen_states:
                seen_states[str(possible_state)] = max_episode
            
        final_decision = random.choice(possible_action_list)
        matrix[final_decision] = 1
        episode_states.append(str(matrix))
        
        if win(matrix):
            winner = "black"
            break
        
        if 0 not in matrix:
            winner = "draw"
            
        else:
            #white move
            possible_action_list = [i for i, v in enumerate(matrix) if v == 0]
            possible_action_expected_reward = [-max_episode]*9
            
            for possible_action in possible_action_list:
                possible_state = matrix.copy()
                possible_state[possible_action] = -1
                
                if str(possible_state) not in seen_states:
                    seen_states[str(possible_state)] = max_episode
                
            action = random.choice(possible_action_list)
            matrix[action] = -1
            episode_states.append(str(matrix))
                
            if win(matrix):
                winner = "white"
                break
            
            if 0 not in matrix:
                winner = "draw"

    #calculate reward for each state observed in this episode
    for state in episode_states:
        
        if winner == "black":
            seen_states[state] += 1
        
        if winner == "white":
            seen_states[state] -= 1
    
    print(np.array(matrix).reshape((3,3)))
    print(winner)
    winning_list.append(winner)
    episode +=1

#plot a graph indicating probability for black to win
x = list(range(1,max_episode+1))
winning_ratio = [int(winning_list[0] == "black")]
for k in range(2, max_episode+1):
    winning_ratio.append(winning_ratio[-1] + (int(winning_list[k-1] == "black") - winning_ratio[-1])/k)
plt.plot(x,winning_ratio)
