from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D


class DQN:
    def __init__(self, shape):
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=shape))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(9, activation='linear'))
        self.model.compile(optimizer='adam', loss="mse")
