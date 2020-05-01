#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ## The Markov Decision Process

# In[2]:


ACTIONS = ["SHOOT", "DODGE", "RECHARGE"]
STAMINA = [0, 50, 100]
ARROWS = [0, 1, 2, 3]
DRAGON_HEALTH = [0, 25, 50, 75, 100]

v_table = np.zeros(shape=(5, 4, 3)) # (arrows, stamina, health)
p_table = np.zeros(shape=(5, 4, 3))
q_table = np.zeros(shape=(5, 4, 3, len(ACTIONS)))


# In[3]:


STATES = [(h, a, s) for h in range(v_table.shape[0]) 
                    for a in range(v_table.shape[1]) 
                    for s in range(v_table.shape[2])]


# In[4]:
INFINTIY = 1e16

def get_next_utility(state: tuple, action: int) -> float:
    """
    Computes the utility of the next state
    """
    assert len(state) == 3 and state[0] < 5 and state[1] < 4 and state[2] < 3 and action < 3
    dragon_health, arrows, stamina = state
    if dragon_health == 0:
        return 0.0
    if ACTIONS[action] == "SHOOT":
        if arrows == 0 or stamina == 0:
            return -INFINTIY
        return 0.5 * v_table[dragon_health, arrows - 1, stamina - 1] + \
            0.5 * v_table[dragon_health - 1, arrows - 1, stamina - 1]
    elif ACTIONS[action] == "DODGE":
        if stamina == 0:
            return -INFINTIY
        elif stamina == 1:
            return 0.8 * v_table[dragon_health, min(arrows + 1, 3), 0] + \
                0.2 * v_table[dragon_health, arrows, 0]
        elif stamina == 2:
            return 0.8 * 0.8 * v_table[dragon_health, min(arrows + 1, 3), 1] + \
                0.2 * 0.8 * v_table[dragon_health, arrows, 1] + \
                0.8 * 0.2 * v_table[dragon_health, min(arrows + 1, 3), 0] + \
                0.2 * 0.2 * v_table[dragon_health, arrows, 0]
    elif ACTIONS[action] == "RECHARGE":
        return 0.8 * v_table[dragon_health, arrows, min(stamina + 1, 2)] + \
            0.2 * v_table[dragon_health, arrows, stamina]


# In[5]:
def get_action_cost(state: tuple, action: int) -> float:
    """
    Returns the reward associated with each action taken
    """
    assert len(state) == 3 and state[0] < 5 and state[1] < 4 and state[2] < 3 and action < 3
    if state[0] == 0:
        return 0.0
    if state[0] == 1 and action == 0:
        return -20.0 + 10.0 * 0.5
    return -20.0 # Penalty = 20 due to Team Number = 9

# In[6]:


