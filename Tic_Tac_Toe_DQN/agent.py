
import random
import numpy as np
from config import config


class Agent:
    def __init__(self, model):
        self.model = model

    def get_action(self, state):
        feasible_action_mask = self.get_feasible_action_mask(state)

        if config.enable_agent:
            action = self.model.forward(state, feasible_action_mask)
        else:
            action = random.choice(feasible_action_mask)

        return action

    def get_feasible_action_mask(self, state):
        mask = np.sum(state, axis=0).reshape(-1) == 0
        return mask

