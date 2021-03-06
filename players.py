import random
from boardgame2 import ReversiEnv
import numpy as np

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
        # Implementar
        # Primero obtengo las acciones válidas
        valid_actions = np.argwhere(self.env.get_valid((board,self.player)))     
        if len(valid_actions)==0:
            print('PASS')
            action = self.env.PASS
        else:
            actions_score = []
            for a in valid_actions:
                (next_state, _), reward, done, _ = self.env.next_step((board, self.player), a)
                actions_score.append(next_state.sum()*self.player)
            best_action_score = max(actions_score)
            best_actions = valid_actions[np.array(actions_score)==best_action_score]
            action = best_actions[random.randint(0,len(best_actions)-1)]
        # Tiene que devoler la acción en la que come más piezas.
        # A igualdad de piezas comidas, samplear uniformemente
        self.env.action_space
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
        # Muestrea aleatoriamente las acciones válidas
        # Puede usar la función creada en la notebook anterior
        #action = sample_valid_actions((board,self.player))
        #print(self.env.get_valid((board,self.player)))
        mat_valid_actions = self.env.get_valid((board,self.player))
        if mat_valid_actions.sum()==0:
            print('PASS')
            action = self.env.PASS
        else:
            valid_actions = np.argwhere(mat_valid_actions) # Acciones validas
            action = valid_actions[random.randint(0,len(valid_actions)-1)]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action
        

class DictPolicyPlayer():
    def __init__(self, player=1, board_shape=4, env=None, flatten_action=False, dict_folder='mdp/pi_mdp.npy'):
        self.pi_dict = np.load(dict_folder, allow_pickle=True).item()
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.player = player
        self.flatten_action = flatten_action
        self.board_shape = board_shape
    
    def predict(self, board):
        # Elegir la acción optima y devolverla
        board_tuple = tuple((board*self.player).reshape(-1))
        if board_tuple in self.pi_dict:
            action = np.array(self.pi_dict[board_tuple])
        else:
            print('PASS')
            action = np.array([-1, 0])
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action