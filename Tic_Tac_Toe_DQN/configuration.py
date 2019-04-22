class Config:
    def __init__(self):
        self.run_mode = None
        self.enable_agent = True
        self.total_train_episode = 100000
        self.intermediate_test_frequency = 1000
        self.number_of_episode_per_intermediate_test = 100
        self.log_frequency = 10000
        self.min_epsilon = 0.2
