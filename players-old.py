from boardgame2 import ReversiEnv
import numpy as np
import gym

class GreedyPlayer():
    def __init__(self, player=1, board_shape=None, env=None, flatten_action=False):
        if (env is None) and (board_shape is None):
            print("board_shape and env can't be both None")
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.env = env
        self.player = player # player number. 1 o -1
        self.flatten_action = flatten_action
        self.board_shape = self.env.board.shape[0]
    
    def predict(self, board):
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            print('pass')
            action = self.env.PASS
        else:
            moves_score = []
            for a in valid_actions:
                next_state, _, _, _ = self.env.next_step((board, self.player), a)
                moves_score.append(next_state[0].sum() * self.player)
            best_score = max(moves_score)
            best_actions = valid_actions[np.array(moves_score)==best_score]
            action = best_actions[np.random.randint(len(best_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action
        
class RandomPlayer():
    def __init__(self, player=1, board_shape=None, env=None, flatten_action=False):
        if (env is None) and (board_shape is None):
            print("board_shape and env can't be both None")
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.env = env
        self.player = player
        self.flatten_action = flatten_action
        self.board_shape = self.env.board.shape[0]
    
    def predict(self, board):
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            print('pass')
            action = self.env.PASS
        else:
            action = valid_actions[np.random.randint(len(valid_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action
        

class DictPolicyPlayer():
    def __init__(self, player=1, board_shape=4, env=None, flatten_action=False, dict_folder='mdp/pi_func_only_winner.npy'):
        self.pi_dict = np.load(dict_folder, allow_pickle=True).item()
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.player = player
        self.flatten_action = flatten_action
        self.board_shape = board_shape
    
    def predict(self, board):
        board_tuple = tuple((board * self.player).reshape(-1))
        action = self.pi_dict[board_tuple]
        if self.flatten_action:
            return action
        else:
            return [action // self.board_shape, action % self.board_shape]