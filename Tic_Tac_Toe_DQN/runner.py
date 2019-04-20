# Import Packages and Modules
from environment import Environment
from agent import Agent
from configuration import Config


class Runner:
    def __init__(self, env, first_mover, second_mover, config):
        self.env = env
        self.first_mover = first_mover
        self.second_mover = second_mover
        self.config = config

    def start(self):
        for train_episode in range(self.config.total_train_episode):
            if train_episode % self.config.intermediate_test_frequency == 0:
                self.config.run_mode = 'test'
                for _ in range(self.config.number_of_episode_per_intermediate_test):
                    self.play_tic_tac_toe(train_episode)
            else:
                self.config.run_mode = 'train'
                self.play_tic_tac_toe(train_episode)

    def play_tic_tac_toe(self, episode):
        epsilon = 1 - episode/self.config.total_train_episode if self.config.run_mode == 'train' else 0
        input_state_of_first_mover, done, info = self.env.reset()
        print(f"episode {episode}: {info}")

        while not done:
            player = 1
            action_of_first_mover = self.first_mover.get_action(input_state_of_first_mover, epsilon)
            input_state_of_second_mover, reward_of_first_mover, done, info = self.env.step(action_of_first_mover, player)
            print(f"episode {episode}: {info}")

            if self.config.run_mode == 'train':
                self.first_mover.store_s_a_r_d(input_state_of_first_mover, action_of_first_mover, reward_of_first_mover, done)
                self.second_mover.store_next_state(input_state_of_second_mover)

                if info is not None:  # Winner is decided after player one takes the move
                    self.first_mover.store_next_state(input_state_of_second_mover)
                    self.second_mover.correct_r(reward_of_first_mover)

            if not done:
                player = 2
                action_of_second_mover = self.second_mover.get_action(input_state_of_second_mover, epsilon)
                input_state_of_first_mover, reward_of_second_mover, done, info = self.env.step(action_of_second_mover, player)
                print(f"episode {episode}: {info}")

                if self.config.run_mode == 'train':
                    self.second_mover.store_s_a_r_d(input_state_of_second_mover, action_of_second_mover, reward_of_second_mover, done)
                    self.first_mover.store_next_state(input_state_of_first_mover)

                    if info is not None:  # Winner is decided after player one takes the move
                        self.second_mover.store_next_state(input_state_of_first_mover)
                        self.first_mover.correct_r(reward_of_second_mover)

        if self.config.run_mode == 'train':
            self.first_mover.train()
            self.second_mover.train()


if __name__ == "__main__":
    env = Environment()
    first_mover = Agent()
    second_mover = Agent()
    config = Config()
    runner = Runner(env, first_mover, second_mover, config)
    runner.start()

