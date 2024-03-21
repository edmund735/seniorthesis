import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    
    metadata = {}

    def __init__(self,
                 render_mode=None,
                 # T = 100.,
                 tau = 10.,
                 lamb = 0.001,
                 gamma = 0.01,
                 sigma = 0.3,
                 c = 0.5,
                 init_pos = 0.,
                 end_pos = 1000.,
                 S0 = 10.):
        
        # self.T = T
        self.lamb = lamb
        self.gamma = gamma
        self.sigma = sigma
        self.c = c
        self.tau = tau
        self.init_pos = init_pos
        self.end_pos = end_pos
        self.S0 = S0 # asset price

        self.observation_space = spaces.Box(low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.]), high = np.array([np.inf, np.inf, np.inf, np.inf, max(self.init_pos, self.end_pos)]), dtype = np.float32)

        self.action_space = spaces.Box(low = 0., high = max(self.init_pos, self.end_pos), shape = (1,), dtype = np.float32)

    def _get_obs(self):
        return np.array([self.J, self.I, self.alpha, self.P, self.Q], dtype = np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self.T_rem = self.T
        self.J = 0. # EWMA of past and current order flow
        self.I = 0. # price impact
        self.alpha = 0. # forecast
        self.S = self.S0 # asset price
        self.P = self.S0 # price of buying with impact
        self.Q = self.init_pos # position

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def step(self, action):

        # Dynamics of price impact
        action = action[0]
        self.J = np.exp(-1/self.tau) * self.J + action
        if self.c == 0.5:
            self.I = self.lamb * np.sign(self.J) * np.sqrt(abs(self.J))
        else:
            self.I = self.lamb * np.sign(self.J) * abs(self.J) ** self.c
        self.alpha = np.exp(-1/self.tau) * self.alpha + self.gamma * self.np_random.standard_normal()
        prev_S = self.S
        self.S += self.alpha + self.sigma * self.np_random.standard_normal()
        self.P = self.S + self.I
        self.Q += action
        # self.T_rem -= 1

        # An episode is done iff the agent has reached the target
        terminated = True if self.Q >= self.end_pos else False
        if self.Q - self.end_pos >= 1:
            reward = -99999
        else:
            reward = -action * self.I + self.Q * (self.S-prev_S)
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
