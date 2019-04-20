# Import Packages and Modules
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

    def start(self):
        for train_episode in range(self.config.total_train_episode):
            if train_episode % self.config.intermediate_test_frequency == 0:
                self.config.run_mode = 'test'
                for test_episode in range(self.config.number_of_episode_per_intermediate_test):
                    self.play_tic_tac_toe(test_episode)
            else:
                self.config.run_mode = 'train'
                self.play_tic_tac_toe(train_episode)

    def play_tic_tac_toe(self, episode):
        epsilon = max(1 - episode/self.config.total_train_episode, self.config.min_epsilon) if self.config.run_mode == 'train' else 0
        state, done, info = self.env.reset()
        player = 0
        print(f"{self.config.run_mode} mode, episode {episode}, {info}")

        while not done:
            action = self.player[player].get_action(state, epsilon)
            self.player[player].store_s_a(state, action)
            state, reward, done, info = self.env.step(action, player)
            self.player[1 - player].store_r_s_d(-reward, state, done)

            if done:
                self.player[player].store_r_s_d(reward, state, done)

            player = 1 - player
            print(f"{self.config.run_mode} mode, episode {episode}: {info}")

        for key in self.player.keys():
            self.player[key].reset()
            self.player[key].train()


if __name__ == "__main__":
    env = Environment()
    config = Config()
    first_mover = Agent(config)
    second_mover = Agent(config)
    runner = Runner(env, first_mover, second_mover, config)
    runner.start()

