import numpy as np

import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, T = 100, lamb = 1, gamma = 0.5, sigma = 0.3, c = 0.5, tau = 1/12, X0 = 1000, P0 = 10):
        
        self.T = T
        self.lamb = lamb
        self.gamma = gamma
        self.sigma = sigma
        self.c = c
        self.tau = tau
        self.P0 = P0 # asset price
        self.X0 = X0 # position

        self.observation_space = spaces.Tuple(spaces.Box(low = -np.inf, high = np.inf),
                                              spaces.Box(low = -np.inf, high = np.inf),
                                              spaces.Box(low = -np.inf, high = np.inf),
                                              spaces.Discrete(self.T_rem),
                                              spaces.Discrete(self.X)
                                              )

        self.action_space = spaces.Discrete(self.X)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        self.T_rem = self.T
        self.J = 0 # EWMA of past and current order flow
        self.I = 0 # price impact
        self.alpha = 0 # forecast
        self.P = self.P # asset price
        self.pi = self.P # price of buying with impact
        self.X = self.X0 # position
        self.rng = np.random.default_rng(seed)

        observation = None # self._get_obs()
        info = None #self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info
    

    def step(self, action):

        # Dynamics of price impact
        self.J = np.exp(-1/self.tau) * self.J + 0.5 * action * (1 + np.exp(-1/(2*self.tau)))
        self.I = self.lamb * np.sign(self.J) * self.J ** self.c
        self.alpha = np.exp(-1/self.tau) * self.alpha + self.gamma * self.rng.standard_normal()
        self.P = self.P + self.alpha + self.sigma * self.rng.standard_normal()
        prev_pi = self.pi
        self.pi = self.P + self.I

        # We use `np.clip` to make sure we don't leave the grid
        self.X = np.clip(self.X - action, 0, self.X)

        # An episode is done iff the agent has reached the target
        terminated = True if self.X == 0 else False

        reward = action * self.pi + self.X * (self.pi-prev_pi)
        observation = None # self._get_obs()
        info = None # self._get_info()

        return observation, reward, terminated, False, info
    

from tqdm import tqdm

env = TradingEnv()

done = False
observation, info = env.reset()