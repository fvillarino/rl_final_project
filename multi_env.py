from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from boardgame2 import ReversiEnv
import gym
import numpy as np

def make_reversi_vec_env(
    env_class,
    n_envs = 1,
    seed = None,
    start_index: int = 0,
    monitor_dir = None,
    wrapper_class = None,
    env_kwargs = None,
    vec_env_cls = None,
    vec_env_kwargs = None,
    monitor_kwargs = None,
    wrapper_kwargs = None,
):
   
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            env = env_class(**env_kwargs)
               
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


class SelfPlayEnv(ReversiEnv):
    def __init__(self, board_shape=8, LocalPlayer=None):
        self.players = [-1, 1]

        self.local_player = LocalPlayer(board_shape=board_shape, flatten_action=False)
        self.board_shape = board_shape
        super(SelfPlayEnv, self).__init__(board_shape=board_shape)
        
        self.action_space = gym.spaces.Discrete(board_shape**2)# Implementar
        self.observation_space = gym.spaces.Box(-1, 1, (1, board_shape,board_shape))# Implementar
         
        
    def play(self, observation):
        action = self.local_player.predict(observation)
        (observation, self.current_player_num), reward, done, info = super(SelfPlayEnv, self).step(action)

        return (observation, self.current_player_num), reward, done, info
    
    def encode_observation(self, observation, valid_actions=False):
        # Implementar
        # Simpre devuelve desde el punto de vista del jugador 1
        # No devuleve el jugador sino solo el tablero
        # Tener en cuenta que esto ser?? la entrada a la red neuronal
        #observation = observation * self.player
        #return observation
        board = observation * self.current_player_num
        if valid_actions:
            return np.array([board, self.get_valid((board, 1))])
        else:
            return observation * self.current_player_num
      
    
    def reset(self):
        self.n_step = 0
        self.local_player_num = np.random.choice(self.players)
        self.local_player.player = self.local_player_num
        self.observation, self.current_player_num = super(SelfPlayEnv, self).reset()
        self.allow_pass = True
            
        if self.current_player_num == self.local_player_num:   
            (self.observation, self.current_player_num), _, done, info = self.play(self.observation)
            assert done == False

        
        return self.encode_observation(self.observation)
    
    def encode_action(self, action):
        # Esta es necesario ya que la salida de la red neuronal ser?? un valor entre 0 y board_shape**2 - 1
        return [action // self.board_shape, action % self.board_shape]
    
    def decode_action(self, action):
        return action[0] * self.board_shape + action[1]

    def step(self, action):
        self.n_step += 1
        action = self.encode_action(action)
        (self.observation, self.current_player_num), reward, done, _ = super(SelfPlayEnv, self).step(action)   
            
        while not done and (self.current_player_num == self.local_player_num):            
            (self.observation, self.current_player_num), reward, done, info = self.play(self.observation)
        
        encoded_observation = self.encode_observation(self.observation)
        
        #print(f'encoded_observation[0]: {encoded_observation[0]}')
        #print(f'encoded_observation[1]: {encoded_observation[1]}')
        
        #reward = float(encoded_observation.sum())# Implementar. Tiene que ser un float
        reward = -float(self.local_player_num*reward)
        
        return encoded_observation, reward, done, {} 