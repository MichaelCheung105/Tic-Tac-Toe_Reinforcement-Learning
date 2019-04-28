import numpy as np
from model import DQN
from experience_pool import ExperiencePool
from configuration import config


class Agent:
    def __init__(self):
        self.eval_net = DQN(shape=(3, 3, 2))
        self.target_net = DQN(shape=(3, 3, 2))
        self.experience_pool = ExperiencePool()
        self.gamma = config.gamma
        self.train_frequency = config.train_frequency
        self.target_net_update_threshold = config.target_net_update_threshold
        self.train_count = 0
        self.target_net_update_count = 0
        self.is_state_stored = False

    def get_action(self, state, epsilon):
        is_random = np.random.rand() < epsilon
        action_list = np.array(range(9))
        model_state = np.expand_dims(state, axis=0)
        q_value_list = self.eval_net.model.predict(model_state)[0] if not is_random else np.random.rand(9)
        feasible_action_mask = self.get_feasible_action_mask(state)
        max_index = np.argmax(q_value_list[feasible_action_mask])
        action = action_list[feasible_action_mask][max_index]
        return action

    @staticmethod
    def get_feasible_action_mask(state):
        mask = np.sum(state, axis=2).reshape(-1) == 0
        return mask

    def store_s_a(self, state, action):
        if config.run_mode == 'train':
            self.experience_pool.store_s_a(state, action)
            self.is_state_stored = True

    def store_r_s_d(self, reward, next_state, done):
        if config.run_mode == 'train':
            if self.is_state_stored:
                self.experience_pool.store_r_s_d(reward, next_state, done)

    def train(self):
        if config.run_mode == 'train':
            if self.experience_pool.storage_count >= self.experience_pool.pool_size and \
                    self.experience_pool.storage_count % self.experience_pool.pool_size == self.train_frequency:
                state, action, reward, next_state, next_action, done = self.experience_pool.sample_experience()
                q_values = self.eval_net.model.predict(state)
                dim_length = range(len(q_values))
                if config.train_method == 'qlearning':
                    next_q_values = np.amax(self.target_net.model.predict(next_state), axis=1)
                elif config.train_method == 'sarsa':
                    next_q_values = self.target_net.model.predict(next_state)[(dim_length, next_action)]
                elif config.train_method == 'ddqn':
                    pass
                else:
                    raise Exception("config.train_method should be one of the following: [qlearning, sarsa, ddqn]")
                target = reward + self.gamma * next_q_values * abs(done - 1)
                q_values[(dim_length, action)] = target
                self.eval_net.model.train_on_batch(state, q_values)
                self.train_count += 1
                print(f"Eval net is trained for {self.train_count} times")

                if self.train_count % self.target_net_update_threshold == 0:
                    self.target_net.model.set_weights(self.eval_net.model.get_weights())
                    self.target_net_update_count += 1
                    print(f"Target net is updated for {self.target_net_update_count} times")

    def reset(self):
        self.is_state_stored = False
