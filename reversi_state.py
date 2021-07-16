from boardgame2 import ReversiEnv
import numpy as np
from copy import deepcopy

class Action():
    def __init__(self, action, player):
        self.action = action
        self.player = player

    def __str__(self):
        return str((self.action[0], self.action[1], self.player))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.action[0] == other.action[0] and self.action[1] == other.action[1] and self.player == other.player

    def __hash__(self):
        return hash((self.action[0], self.action[1]))
    
class ReversiState():
    def __init__(self, board_shape=6):
        self.env = ReversiEnv(board_shape=board_shape)
        self.board, self.currentPlayer = self.env.reset()
        self.done = False
    
    def getCurrentPlayer(self):
        return self.currentPlayer
    
    def get_actions_mask(self):
        return self.env.get_valid((self.board, self.currentPlayer)).reshape(-1)

    def getPossibleActions(self):
        possibleActions = []
#         valid_actions = np.argwhere(self.board == 0)
        valid_actions = np.argwhere(self.env.get_valid((self.board, self.currentPlayer)))
        for action in valid_actions:
            possibleActions.append(Action(action, self.currentPlayer))
        return possibleActions
    
    def takeAction(self, action):
        newState = deepcopy(self)
        (newState.board, newState.currentPlayer), newState.reward, newState.done, info = self.env.next_step((self.board, self.currentPlayer), (action.action[0], action.action[1]))
        return newState
    
    def isTerminal(self):
        return self.done
    
    def getReward(self):
        return self.reward
    
    def encode_state(self):
        matrix = np.ones(self.board.shape + (3,))
        matrix[:,:,0] = (self.board == 1) * 1.0
        matrix[:,:,1] = (self.board == -1) * 1.0
#         matrix[:,:,2] = self.env.get_valid((self.board, self.currentPlayer))
        matrix[:,:,2] = matrix[:,:,2] * (self.currentPlayer == 1)

        observation = matrix.transpose((2, 0, 1))
        return observation