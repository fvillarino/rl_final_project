import numpy as np

def generate_uniform_stochastic_policy(states_actions):
    # Given the states generate a uniform stochastic policy
    pi = {}
    for state, actions in states_actions.items():
        pi[state] = {}
        prob = 1/len(actions)
        for action, data in actions.items():
            pi[state][action] = prob
    return pi

def generate_deterministic_policy(states_actions):
    # Chooses a random action from all action in state
    pi = {}
    for state, actions in states_actions.items():
        pi[state] = list(actions.keys())[np.random.choice(len(actions.keys()))] # np.random.choice(list(actions.keys()))
    return pi

def stochastic_policy_eval_step(states_actions, V, pi):
    # Evaluation in place (in contrast with evaluation with 2 arrays).
    # Needs less memory and converges too
    # pi is a dict and pi[s] is a dict too. 
    # pi[s][a] is the probability of an action given the state
    delta = 0
    for state, actions in states_actions.items():
        V_updated = 0
        for action, data in actions.items():
            next_node = data['next_node']
            reward = data['winner']
            done = data['done']
            prob = pi[state][action]
            if not done:
                V_updated = V_updated + prob*(- V[next_node])
            else:
                # Esto es un nodo terminal
                V_updated = V_updated - prob*reward
        delta = max(delta, np.abs(V_updated - V[state]))
        V[state] = V_updated
    return V, delta

def deterministic_policy_eval_step(states_actions, V, pi):
    # Evaluation in place (in contrast with evaluation with 2 arrays).
    # Needs less memory and converges too
    # pi is a dict and pi[s] is the best action for that state. (The most probable action)
    delta = 0
    for state, actions in states_actions.items():
        V_updated = 0
        action = pi[state]
        next_node = actions[action]['next_node']
        done = actions[action]['done']
        reward = actions[action]['winner']
        if not done:
            V_updated = V_updated + (-V[next_node])
        else:
            # Esto es un nodo terminal
            V_updated = V_updated - reward
        delta = max(delta, np.abs(V_updated - V[state]))
        V[state] = V_updated
    return V, delta

def policy_evaluation(policy_eval_step, states_actions, pi, theta, verbose=0):
    if verbose:
        print('Iteration number: ', end=' ')
    
    V = {}
    iters = 0
    for state in states_actions:
        V[state] = 0
    delta = theta + 1
    while theta<delta: 
        V, delta = policy_eval_step(states_actions, V, pi)
        iters += 1
        if verbose:
            print(iters, end=' ')
    print()
    return V, iters

def policy_improve(V, states_actions):
    pi = {}
    for state, actions in states_actions.items():
        actions_list = list(actions.keys())
        expected_rewards = np.zeros(len(actions_list))
        for i, (action, data) in enumerate(actions.items()):
            next_state = data['next_node']
            reward = data['winner']
            done = data['done']
            if not done:
                expected_rewards[i] = - V[next_state]
            else:
                # Esto es un nodo terminal
                expected_rewards[i] = - reward
        pi[state] = actions_list[np.argmax(expected_rewards)]
    return pi

def policy_iteration(states_actions, pi_old, deterministic_policy_eval_step = deterministic_policy_eval_step, policy_improve=policy_improve, verbose = 0):
    # Politica inicial
    policy_updates = 100
    while policy_updates > 0:
        # Calculo values de politica
        V, iters = policy_evaluation(deterministic_policy_eval_step, states_actions, pi_old, 1e-6, verbose=verbose)
        # Mejoro política con values
        pi = policy_improve(V, states_actions)

        policy_updates = 0
        for j, (state, accion) in enumerate(pi.items()):
            if accion != pi_old[state]:
                 policy_updates += 1
        pi_old = pi.copy()
        if verbose:
            print('Number of differences of new policy vs old policy:', policy_updates)
            print('---------------------------')
    return pi_old, V

def value_iteration(states_actions, theta=1e-8, verbose=0):
    V = {}
    iters = 0
    for state in states_actions:
        V[state] = 0
    delta = theta + 1
    iterat = 0
    N = len(states_actions)
    while theta<delta: 
        suma = 0
        delta = 0
        for state, actions in states_actions.items():
            expected_rewards = []
            for action in actions:
                next_state = actions[action]['next_node']
                winner = actions[action]['winner']
                done = actions[action]['done']
                reward = np.array(next_state).sum()
                if not done:
                    expected_rewards.append(-V[next_state])
                else:
                    # Esto es un nodo terminal
                    expected_rewards.append(-reward)
            V_updated = max(expected_rewards)
            suma = suma + np.abs(V_updated - V[state])
            delta = max(delta, np.abs(V_updated - V[state]))
            V[state] = V_updated
        iterat += 1
        if verbose == 1:
            print(iterat, delta, suma/N)
    return V, delta