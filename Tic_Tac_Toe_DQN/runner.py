# Import Packages and Modules
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from environment import Environment
from agent import Agent
from configuration import Config


class Runner:
    def __init__(self, env, first_mover, second_mover, config):
        self.env = env
        self.config = config
        self.player = {0: first_mover,
                       1: second_mover
                       }
        self.trained_episode = 0
        self.logs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    def start(self):
        for train_episode in range(1, self.config.total_train_episode + 1):
            if train_episode % self.config.intermediate_test_frequency == 0:
                self.config.run_mode = 'test'
                for test_episode in range(self.config.number_of_episode_per_intermediate_test):
                    self.play_tic_tac_toe(test_episode, "both_greedy")
                    self.play_tic_tac_toe(test_episode, "first_player_random")
                    self.play_tic_tac_toe(test_episode, "second_player_random")
            else:
                self.config.run_mode = 'train'
                self.play_tic_tac_toe(train_episode)
                self.trained_episode = train_episode

            if train_episode % self.config.log_frequency == 0:
                self.log_result(train_episode)
                # self.log_experience_pool(train_episode)

        for key, player in self.player.items():
            if not os.path.exists('./models'):
                os.makedirs('./models')
            player.eval_net.model.save(f"./models/player_{key}.h5")

    def play_tic_tac_toe(self, episode, test_condition=None):
        state, done, info, player = self.env.reset()
        print(f"{self.config.run_mode} mode, episode {episode}, {info}")

        while not done:
            player = 1 - player
            epsilon = self.determine_epsilon(episode, player, test_condition)
            action = self.player[player].get_action(state, epsilon)
            self.player[player].store_s_a(state, action)
            state, reward, done, info = self.env.step(action, player)
            self.player[1 - player].store_r_s_d(-reward, state, done)

            if done:
                self.player[player].store_r_s_d(reward, state, done)

            print(f"{self.config.run_mode} mode, episode {episode}: {info}")

        for key, agent in self.player.items():
            agent.reset()
            agent.train()

        if self.config.run_mode == 'test':
            self.record_result_of_first_player(player, reward, test_condition)

    def determine_epsilon(self, episode, player, test_condition):
        if player == 0 and test_condition == "first_player_random":
            epsilon = 1
        elif player == 1 and test_condition == "second_player_random":
            epsilon = 1
        else:
            epsilon = self.default_epsilon(episode)
        return epsilon

    def default_epsilon(self, episode):
        epsilon = max(1 - episode / self.config.total_train_episode, self.config.min_epsilon) if self.config.run_mode == 'train' else 0
        return epsilon

    def record_result_of_first_player(self, player, reward, test_condition):
        reward = reward if player == 0 else -reward
        self.logs['result'][test_condition][self.trained_episode] += reward

    def log_result(self, episode):
        for test_condition, records in self.logs.items():
            df = pd.DataFrame.from_dict(records)
            df.plot(title=test_condition)
            if not os.path.exists('./result'):
                os.makedirs('./result')
            plt.savefig(f"./result/{test_condition}_Ep{episode}.jpg")
            plt.close()

    def log_experience_pool(self, episode):
        for key, agent in self.player.items():
            states = agent.experience_pool.state
            actions = agent.experience_pool.action
            rewards = agent.experience_pool.reward
            next_states = agent.experience_pool.next_state
            done = agent.experience_pool.done


if __name__ == "__main__":
    env = Environment()
    config = Config()
    first_mover = Agent(config)
    second_mover = Agent(config)
    runner = Runner(env, first_mover, second_mover, config)
    runner.start()

