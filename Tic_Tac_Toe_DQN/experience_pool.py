import numpy as np


class ExperiencePool:
    def __init__(self, pool_size=1000, batch_size=200):
        self.state = [None] * pool_size
        self.action = [None] * pool_size
        self.reward = [None] * pool_size
        self.next_state = [None] * pool_size
        self.done = [None] * pool_size
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.storage_count = 0

    def sample_experience(self):
        sample_index = np.random.randint(0, self.pool_size, self.batch_size)
        sample_state = np.array(self.state)[sample_index]
        sample_action = np.array(self.action)[sample_index]
        sample_reward = np.array(self.reward)[sample_index]
        sample_next_state = np.array(self.next_state)[sample_index]
        sample_done = np.array(self.done)[sample_index]
        return sample_state, sample_action, sample_reward, sample_next_state, sample_done

    def store_s_a(self, state, action):
        index = self.storage_count % self.pool_size
        self.state[index] = state.copy()
        self.action[index] = action

    def store_r_s_d(self, reward, next_state, done):
        index = self.storage_count % self.pool_size
        self.reward[index] = reward
        self.done[index] = done
        self.next_state[index] = next_state.copy()
        self.storage_count += 1
