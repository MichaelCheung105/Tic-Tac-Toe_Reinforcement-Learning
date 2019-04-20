import numpy as np


class ExperiencePool:
    def __init__(self, pool_size=1000, batch_size=100):
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
        sample_state = self.state[sample_index]
        sample_action = self.action[sample_index]
        sample_reward = self.reward[sample_index]
        sample_next_state = self.next_state[sample_index]
        sample_done = self.done[sample_index]
        return sample_state, sample_action, sample_reward, sample_next_state, sample_done

    def store_s_a_r_d(self, state, action, reward, done):
        index = self.storage_count % self.pool_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.done[index] = done
        return index

    def store_next_state(self, next_state):
        index = self.storage_count % self.pool_size
        self.next_state[index] = next_state
        self.storage_count += 1
