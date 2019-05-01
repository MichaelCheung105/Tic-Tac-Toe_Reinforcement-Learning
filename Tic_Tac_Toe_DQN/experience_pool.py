import numpy as np
from configuration import config
from rl_logging import logs


class ExperiencePool:
    def __init__(self, player):
        self.player = player
        self.state = [None] * config.pool_size
        self.action = [None] * config.pool_size
        self.reward = [None] * config.pool_size
        self.next_state = [None] * config.pool_size
        self.next_action = [None] * config.pool_size
        self.done = [None] * config.pool_size
        self.pool_size = config.pool_size
        self.batch_size = config.batch_size
        self.storage_count = 0
        self.sampled_count = 0

    def sample_experience(self):
        self.sampled_count += 1
        sample_index = np.random.randint(0, self.pool_size, self.batch_size)
        sample_state = np.array(self.state)[sample_index]
        sample_action = np.array(self.action)[sample_index]
        sample_reward = np.array(self.reward)[sample_index]
        sample_next_state = np.array(self.next_state)[sample_index]
        sample_next_action = np.array(self.next_action)[sample_index]
        sample_done = np.array(self.done)[sample_index]
        self.record_statistics_of_sampled_experience(sample_reward, sample_state)
        return sample_state, sample_action, sample_reward, sample_next_state, sample_next_action, sample_done

    def record_statistics_of_sampled_experience(self, sample_reward, sample_state):
        unique_sampled_states = set([str(s) for s in sample_state])
        logs['sampled_experience']["player_{0}".format(self.player)]['cumulative_info']['cumulative_seen_states'].update(unique_sampled_states)
        sampled_experience_dict = logs['sampled_experience']["player_{0}".format(self.player)]['per_sampling_info'][self.sampled_count]
        sampled_experience_dict['num_of_states'] = len(sample_state)
        sampled_experience_dict['num_of_unique_seen_states'] = len(unique_sampled_states)
        sampled_experience_dict['num_of_won'] = sum(sample_reward > 0)
        sampled_experience_dict['num_of_draw'] = sum(sample_reward == 0)
        sampled_experience_dict['num_of_lose'] = sum(sample_reward < 0)
        sampled_experience_dict['cumulative_num_of_seen_states'] = \
            len(logs['sampled_experience']["player_{0}".format(self.player)]['cumulative_info'][
                    'cumulative_seen_states'])

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
