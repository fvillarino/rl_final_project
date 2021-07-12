from boardgame2 import ReversiEnv
import collections
import numpy as np

def get_valid_modes(state, env):
    valid_moves = env.get_valid((state, 1))
    valid_moves = np.argwhere(valid_moves == 1)
    if len(valid_moves) == 0:
        valid_moves = [env.PASS]
    return valid_moves

def bfs_cannonical(board_shape, verbose=0):
    env = ReversiEnv(board_shape=board_shape)
    (board, first_player) = env.reset()
    state_tuple = tuple(first_player * board.reshape(-1))
    seen = set([])
    cannonical_states = {}
    # deque es como una pila pero con doble entrada (rear-front)
    queue = collections.deque([state_tuple])
    while queue:
        vertex = queue.popleft()
        state = np.array(vertex).reshape(board_shape, board_shape)
        valid_moves = get_valid_modes(state, env)
        if env.get_winner((state, 1)) is None:
            cannonical_states[vertex] = {}
            for action in valid_moves:
                action = tuple(action)
                cannonical_states[vertex][action] = {}
                (next_state, _), reward, done, _ = env.next_step((state, 1), action)

                next_state = next_state * -1 # Cannonical Form
                node = tuple(next_state.reshape(-1))
                cannonical_states[vertex][action]['done'] = done
                cannonical_states[vertex][action]['winner'] = -1 * reward
                cannonical_states[vertex][action]['next_node'] = node
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        if verbose==1:
            print(f'{len(cannonical_states)}\r', end='')
        
    return cannonical_states