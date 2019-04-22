import numpy as np


class Environment:
    def __init__(self):
        self.board = None

    def reset(self):
        self.board = np.zeros(shape=(3, 3, 2))
        done = False
        info = "New Game, Player 1's Turn"
        player = 1
        return self.board, done, info, player

    def step(self, action, player):
        self.board.reshape((9, 2))[action, player] = 1
        reward, done, info = self.check_status(self.board, player)
        return self.board, reward, done, info

    def check_status(self, board, player):
        done, is_won = self.check_if_done(board, player)
        reward, info = 0, f"Player {2-player}'s Turn"

        if is_won:
            reward = 1
            info = f"The winner is Player {player + 1}"
        elif done:
            info = "Draw"

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

    def check_if_done(self, board, player):
        is_won = self.check_winner(board[:, :, player])
        done = 0 not in np.sum(board, axis=2) or is_won
        return done, is_won
