
import numpy as np

class Environment:
    def __init__(self):
        self.board = None

    def reset(self):
        self.board = np.zeros(shape=(2, 3, 3))
        done = False
        return self.board, done

    def step(self, action):
        self.board[action] = 1
        reward, done, info = self.check_winner(self.board)
        return self.board, reward, done, info

    def check_winner(self, board):
        winner = self.check_score(board)
        if bool(winner):
            reward, done = 1, True
            info = '{0} wins!'.format(winner)
        else:
            reward, done, info = 0, False, None
        return reward, done, info

    def check_score(self, board):
        winner = 1 in np.sum(board, axis=1)
        return winner

