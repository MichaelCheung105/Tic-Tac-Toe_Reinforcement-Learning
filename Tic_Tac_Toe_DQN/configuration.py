class Config:
    def __init__(self):
        self.run_mode = None
        self.enable_agent = True
        self.total_train_episode = 100000
        self.intermediate_test_frequency = 1000
        self.number_of_episode_per_intermediate_test = 100
        self.log_frequency = 10000

        # Experience Pool
        self.pool_size = 1000
        self.batch_size = 200

        # Agent
        self.min_epsilon = 0.2
        self.gamma = 0.9
        self.train_frequency = 100
        self.target_net_update_threshold = 10
        self.train_method = 'sarsa'  # qlearning, sarsa or ddqn


config = Config()
