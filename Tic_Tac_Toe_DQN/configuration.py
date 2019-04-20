class Config:
    def __init__(self):
        self.run_mode = None
        self.enable_agent = True
        self.total_train_episode = 1000
        self.intermediate_test_frequency = 100
        self.number_of_episode_per_intermediate_test = 10
        self.min_epsilon = 0.3
