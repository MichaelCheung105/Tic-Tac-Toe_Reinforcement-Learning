import numpy as np
from model import DQN
from experience_pool import ExperiencePool


class Agent:
    def __init__(self):
        self.eval_net = DQN(shape=(2, 3, 3))
        self.target_net = DQN(shape=(2, 3, 3))
        self.experience_pool = ExperiencePool()
        self.gamma = 0.9
        self.train_threshold = 100
        self.train_count = 0
        self.target_net_update_threshold = 10
        self.last_storage_index = 0

    def get_action(self, state, epsilon):
        is_random = np.random.rand() < epsilon
        q_value_list = self.eval_net.model.predict(state) if not is_random else np.random.rand(9)
        feasible_action_mask = self.get_feasible_action_mask(state)
        max_q = max(q_value_list[feasible_action_mask])
        action = list(q_value_list).index(max_q)
        return action

    @staticmethod
    def get_feasible_action_mask(state):
        mask = np.sum(state, axis=0).reshape(-1) == 0
        return mask

    def store_s_a_r_d(self, state, action, reward, done):
        index = self.experience_pool.store_s_a_r_d(state, action, reward, done)
        self.last_storage_index = index

    def store_next_state(self, next_state):
        self.experience_pool.store_next_state(next_state)

    def train(self):
        if self.experience_pool.storage_count >= self.experience_pool.pool_size and \
                self.experience_pool.storage_count % self.experience_pool.pool_size == self.train_threshold:
            state, action, reward, next_state, done = self.experience_pool.sample_experience()
            q_values = self.eval_net.model.predict(state)
            next_q_values = self.target_net.model.predict(next_state)
            target = reward + self.gamma * next_q_values * (abs(done - 1))
            q_values[(range(len(q_values)), action)] = target
            self.eval_net.model.train_on_batch(state, q_values)
            self.train_count += 1

        if self.train_count == self.target_net_update_threshold:
            self.target_net.model.set_weights(self.eval_net.model.get_weights())
            self.train_count = 0

    def correct_r(self, reward):
        self.experience_pool.reward[self.last_storage_index] = -reward

