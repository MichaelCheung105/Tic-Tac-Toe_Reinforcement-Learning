
# Import Packages and Modules
from environment import Environment
from agent import Agent
from model import Model
from config import config

class Runner:
    def __init__(self, env, first_mover, second_mover):
        self.env = env
        self.first_mover = first_mover
        self.second_mover = second_mover

    def start(self):
        for train_episode in range(config.total_train_episode):
            if train_episode % config.intermediate_test_frequency == 0:
                config.run_mode = 'test'
                for _ in range(config.number_of_episode_per_intermediate_test):
                    self.play_tic_tac_toe()
            else:
                config.run_mode = 'train'
                self.play_tic_tac_toe()

    def play_tic_tac_toe(self):
        input_state_of_first_mover, done = self.env.reset()

        while not done:
            action_of_first_mover = self.first_mover.get_action(input_state_of_first_mover)
            input_state_of_second_mover, reward_of_first_mover, done, info = self.env.step(action_of_first_mover)

            if config.run_mode == 'train':
                self.first_mover.store_s_a_r(input_state_of_first_mover, action_of_first_mover, reward_of_first_mover)
                self.second_mover.store_next_state(input_state_of_second_mover)

            if not done:
                action_of_second_mover = self.second_mover.get_action(input_state_of_second_mover)
                input_state_of_first_mover, reward_of_second_mover, done, info = self.env.step(action_of_second_mover)

                if config.run_mode == 'train':
                    self.second_mover.store_s_a_r(input_state_of_second_mover, action_of_second_mover, reward_of_second_mover)
                    self.first_mover.store_next_state(input_state_of_first_mover)

        if config.run_mode == 'train':
            self.first_mover.train()
            self.second_mover.train()


if __name__ == "__main__":
    env = Environment()
    first_mover = Agent(Model)
    second_mover = Agent(Model)
    runner = Runner(env, first_mover, second_mover)
    runner.start()

