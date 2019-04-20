import numpy as np


class Environment:
    def __init__(self):
        self.board = None

    def reset(self):
        self.board = np.zeros(shape=(2, 3, 3))
        done = False
        info = "New Game"
        return self.board, done, info

    def step(self, action, player):
        index = player - 1
        self.board[index][action] = 1
        reward, done, info = self.check_status(self.board, index)
        return self.board, reward, done, info

    def check_status(self, board, index):
        is_won = self.check_winner(board[index])
        done = self.check_if_done(board)

        if is_won:
            reward = 1
            info = f"The winner is player {index + 1}"
        else:
            reward = 0
            info = None

        return reward, done, info

    @staticmethod
    def check_winner(board):
        check_list = []
        check_list.extend(np.sum(board, axis=0))
        check_list.extend(np.sum(board, axis=1))
        check_list.append(np.trace(board))
        check_list.append(np.trace(np.fliplr(board)))
        is_won = 3 in check_list
        return is_won

    @staticmethod
    def check_if_done(board):
        is_done = 0 not in np.sum(board, axis=0)
        return is_done
