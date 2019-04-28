import numpy as np
from configuration import config


class ExperiencePool:
    def __init__(self):
        self.state = [None] * config.pool_size
        self.action = [None] * config.pool_size
        self.reward = [None] * config.pool_size
        self.next_state = [None] * config.pool_size
        self.next_action = [None] * config.pool_size
        self.done = [None] * config.pool_size
        self.pool_size = config.pool_size
        self.batch_size = config.batch_size
        self.storage_count = 0

    def sample_experience(self):
        sample_index = np.random.randint(0, config.pool_size, config.batch_size)
        sample_state = np.array(self.state)[sample_index]
        sample_action = np.array(self.action)[sample_index]
        sample_reward = np.array(self.reward)[sample_index]
        sample_next_state = np.array(self.next_state)[sample_index]
        sample_next_action = np.array(self.next_action)[sample_index]
        sample_done = np.array(self.done)[sample_index]
        return sample_state, sample_action, sample_reward, sample_next_state, sample_next_action, sample_done

    def store_s_a(self, state, action):
        current_index = self.storage_count % config.pool_size
        self.state[current_index] = state.copy()
        self.action[current_index] = action

        if self.storage_count != 0:
            last_index = (self.storage_count - 1) % config.pool_size
            self.next_action[last_index] = action

    def store_r_s_d(self, reward, next_state, done):
        index = self.storage_count % config.pool_size
        self.reward[index] = reward
        self.done[index] = done
        self.next_state[index] = next_state.copy()
        self.storage_count += 1
