import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    
    metadata = {}

    def __init__(self,
                 render_mode=None,
                 T = 100.,
                 impact_decay = 10.,
                 lamb = 0.001,
                 gamma = 0.01,
                 sigma = 0.3,
                 c = 0.5,
                 target_q = 1000,
                 S0 = 10,
                 ):
        
        assert T > 0, "T must be positive"
        assert impact_decay > 0, "impact decay must be positive"
        assert lamb > 0, "lambda must be positive"
        assert 0 < c <= 1, "concavity must be in (0,1]"
        assert target_q > 0, "Target quantity must be positive"
        assert S0 > 0, "Initial price must be positive"

        self.T = T
        self.lamb = lamb
        self.gamma = gamma
        self.sigma = sigma
        self.c = c
        self.impact_decay = impact_decay
        self.target_q = target_q
        self.S0 = S0 # asset price

        # for low and high, state variables are J, I, alpha, P, Q_rem, T_rem
        low = np.array([0, 0, -np.inf, 0, 0, 0])
        high = np.array([target_q, lamb * target_q ** c, np.inf, np.inf, target_q, T])
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float64)

        self.action_space = spaces.Box(low = 0, high = target_q, shape = (1,), dtype = np.float64)

    def _get_obs(self):
        if self.c == 0.5:
            I = self.lamb * np.sign(self.J) * np.sqrt(abs(self.J))
        else:
            I = self.lamb * np.sign(self.J) * abs(self.J) ** self.c
        P = self.S + I
        return np.array([self.J, I, self.alpha, P, self.target_q - self.Q, self.T_rem], dtype = np.float64)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.T_rem = self.T
        self.J = 0. # EWMA of past and current order flow
        self.alpha = 0. # forecast
        self.S = self.S0 # asset price
        self.Q = 0. # position

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def step(self, action):

        # Dynamics of price impact
        action = action[0]
        self.J = np.exp(-1/self.impact_decay) * self.J + action
        self.alpha = np.exp(-1/self.impact_decay) * self.alpha + self.gamma * self.np_random.standard_normal()
        prev_S = self.S
        self.S += self.alpha + self.sigma * self.np_random.standard_normal()
        self.Q += action
        self.T_rem -= 1

        # An episode is terminated if the current quantity is above the target or there's no more time
        terminated = True if self.Q >= self.target_q or self.T_rem <= 0 else False

        observation = self._get_obs()

        # if bought too many shares
        if self.Q - self.target_q >= 0.1:
            reward = -np.inf
        else:
            reward = -action * observation[1] + self.Q * (self.S-prev_S)
        
        info = self._get_info()

        return observation, reward, terminated, False, info
