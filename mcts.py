from __future__ import division

import time
import math
import random
import numpy as np
from reversi_state import Action

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


def modelPolicy(model):
    def sampleModel(state):
        while not state.isTerminal():
            try:
                action_probs = model.eval()(state.board.reshape(1, 1, *state.board.shape)).detach().numpy()[0] * state.get_actions_mask()
                action_probs = action_probs/action_probs.sum()
                action = np.random.choice(len(action_probs), p=action_probs)
                coded_action = [action % state.board.shape[0], action // state.board.shape[0]]
                action = Action(coded_action , state.getCurrentPlayer())
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            state = state.takeAction(action)
        return state.getReward()
    return sampleModel


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        
        if needDetails:
            node_actions, bestChild = self.getBestChild(self.root, 0, verbose=needDetails)
            #return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
            action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
            return node_actions, action
        else:
            bestChild = self.getBestChild(self.root, 0, verbose=needDetails)
            action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
#         print(node.state.board)
#         print(node.isTerminal)
#         print('rollout')
        reward = self.rollout(node.state)
#         print('backpropogate')
        self.backpropogate(node, reward)
#         print()

    def selectNode(self, node):
        while not node.isTerminal:
#             print('not terminal')
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
#                 print('not isFullyExpanded')
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
#             print(action)
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
#                 print('descubre nodo')
                node.children[action] = newNode
                if len(actions) == len(node.children):
#                     print('fullyexpanded')
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue, verbose=0):
        bestValue = float("-inf")
        bestNodes = []
        node_actions = {}
#         total_visits = 0
#         total_reward = 0
        for key, child in node.children.items():
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
#             total_visits += child.numVisits
#             total_reward += child.totalReward * node.state.getCurrentPlayer()
            if verbose:
                node_actions[(key.action[0], key.action[1], key.player)] = (nodeValue, child.totalReward, child.numVisits, (child.totalReward + child.numVisits)/2/child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        if verbose:
#             print(node_actions)
#             print(total_visits, total_reward)
            return node_actions, random.choice(bestNodes)
        return random.choice(bestNodes)