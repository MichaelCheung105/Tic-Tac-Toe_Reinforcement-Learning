# Import Packages and Modules
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from environment import Environment
from agent import Agent
from configuration import config
from rl_logging import logs


class Runner:
    def __init__(self, env, first_mover, second_mover):
        self.env = env
        self.player = {0: first_mover,
                       1: second_mover
                       }
        self.trained_episode = 0

    def start(self):
        for train_episode in range(1, config.total_train_episode + 1):
            if train_episode % config.intermediate_test_frequency == 0:
                config.run_mode = 'test'
                for test_episode in range(config.number_of_episode_per_intermediate_test):
                    self.play_tic_tac_toe(test_episode, "both_greedy")
                    self.play_tic_tac_toe(test_episode, "first_player_random")
                    self.play_tic_tac_toe(test_episode, "second_player_random")
                    self.play_tic_tac_toe(test_episode, "both_random")
            else:
                config.run_mode = 'train'
                self.play_tic_tac_toe(train_episode)
                self.trained_episode = train_episode

            if train_episode % config.log_frequency == 0:
                self.export_logs(train_episode)
                # self.log_experience_pool(train_episode)  #TODO: complete the logging of experience pool

        for key, player in self.player.items():
            directory_path = f"./models/{config.experiment_name}"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            output_path = os.path.join(directory_path, f"player_{key+1}.h5")
            player.eval_net.model.save(output_path)

    def play_tic_tac_toe(self, episode, test_condition=None):
        state, reward, done, info, player = self.env.reset()
        print(f"{config.run_mode} mode, episode {episode}, {info}")

        while not done:
            player = 1 - player
            epsilon = self.determine_epsilon(episode, player, test_condition)
            action = self.player[player].get_action(state, epsilon)
            self.player[player].store_s_a(state, action)
            state, reward, done, info = self.env.step(action, player)
            self.player[1 - player].store_r_s_d(-reward, state, done)

            if done:
                self.player[player].store_r_s_d(reward, state, done)

            print(f"{config.run_mode} mode, episode {episode}: {info}")

        for key, agent in self.player.items():
            agent.reset()
            agent.train()

        if config.run_mode == 'test':
            assert reward is not None
            self.record_result_of_players(player, reward, test_condition)

    def determine_epsilon(self, episode, player, test_condition):
        if player == 0 and test_condition == "first_player_random":
            epsilon = 1
        elif player == 1 and test_condition == "second_player_random":
            epsilon = 1
        elif test_condition == "both_random":
            epsilon = 1
        else:
            epsilon = self.default_epsilon(episode)
        return epsilon

    @staticmethod
    def default_epsilon(episode):
        epsilon = max(1 - episode / config.total_train_episode, config.min_epsilon) if config.run_mode == 'train' else 0
        return epsilon

    def record_result_of_players(self, player, reward, test_condition):
        # Record how many times the each player loses the game
        losses_of_first_player = -reward if player == 1 else 0
        logs['result']["player_{0}".format(1)][test_condition][self.trained_episode] += losses_of_first_player
        losses_of_second_player = -reward if player == 0 else 0
        logs['result']["player_{0}".format(2)][test_condition][self.trained_episode] += losses_of_second_player

    def export_logs(self, episode):
        for log_category, log_content in logs.items():
            if log_category == 'result':
                self.log_results(episode, log_category, log_content)
            else:
                pass

    @staticmethod
    def log_results(episode, log_category, log_content):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,60))
        plot_index = 0
        for player, records in log_content.items():
            df = pd.DataFrame.from_dict(records)
            df.plot(title=f"# of losses of {player}", grid=True, ax=axes[plot_index])
            plt.xlabel("# of trained episodes")
            plt.ylabel("Total number of losses")
            plot_index += 1
        directory_path = f"./result/{config.experiment_name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        output_path = os.path.join(directory_path, f"{log_category}_Ep{episode}.jpg")
        plt.savefig(output_path)
        plt.close()

    def log_experience_pool(self, episode):
        #TODO: Complete logging of experience pool
        experience_pool_dict = defaultdict(lambda: {})
        for key, agent in self.player.items():
            player_dict = experience_pool_dict[f"player_{key+1}"]
            player_dict["states"] = agent.experience_pool.state
            player_dict["actions"] = agent.experience_pool.action
            player_dict["rewards"] = agent.experience_pool.reward
            player_dict["next_states"] = agent.experience_pool.next_state
            player_dict["next_actions"] = agent.experience_pool.next_actions
            player_dict["done"] = agent.experience_pool.done


if __name__ == "__main__":
    env = Environment()
    first_mover = Agent()
    second_mover = Agent()
    runner = Runner(env, first_mover, second_mover)
    runner.start()
