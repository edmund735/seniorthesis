import numpy as np
import gymnasium as gym
from gymnasium import spaces, logger


class TradingEnv(gym.Env):
    
    metadata = {}

    def __init__(self,
                 render_mode=None,
                 T = 100.,
                 impact_decay = 10.,
                 lamb = 0.01,
                 gamma = 1,
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
        self.T_rem = None
        self.impact_decay = impact_decay
        self.lamb = lamb
        self.gamma = gamma
        self.sigma = sigma
        self.c = c
        self.target_q = target_q
        self.S0 = S0
        # self.bench_x = self.target_q / self.

        # for low and high, state variables are J, I, alpha, P, Q_rem, T_rem
        low = np.array([0, 0, -np.inf, -np.inf, 0, 0])
        high = np.array([target_q, lamb * np.sign(target_q) * abs(target_q) ** c, np.inf, np.inf, target_q, T])
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float64)

        # Actions will be rescaled in step function
        self.action_space = spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.float32)

    def _get_obs(self):
        if self.c == 0.5:
            I = self.lamb * np.sign(self.J) * np.sqrt(abs(self.J))
        else:
            I = self.lamb * np.sign(self.J) * abs(self.J) ** self.c
        P = self.S + I
        return np.array([self.J, I, self.alpha, P, self.Q_rem, self.T_rem], dtype = np.float64)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.T_rem = self.T
        self.J = 0. # EWMA of past and current order flow
        self.alpha = 0. # forecast
        self.S = self.S0 # asset price
        self.Q_rem = self.target_q # position

        # Benchmark
        # self.bench_J = 0.
        # self.bench_alpa = 0.
        # self.bench_S = self.S0
        # self.Q = 0.

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def step(self, action):

        assert self.T_rem is not None, "You must call reset() before step()"
        if self.T_rem <= 0:
            logger.warn("This episode has already terminated. Please call reset() before step()")
        if not self.action_space.contains(action):
            if self.action_space.contains(action.astype(np.float32)):
                logger.warn(f"Actions should be of type np.float32 but are currently {type(action[0])}")
            else:
                assert self.action_space.contains(action), f"The action {action!r} ({type(action)}) is invalid (actions must be in [-1,1])"

        # Passed in action is in [-1,1], need to rescale to [0,Q_rem]
        x = (action[0] + 1)/2*self.Q_rem

        # Must reach target_q by last time step
        if self.T_rem <= 1:
            x = self.Q_rem

        # Dynamics of price impact
        self.J = np.exp(-1/self.impact_decay) * self.J + x
        self.alpha = np.exp(-1/self.impact_decay) * self.alpha + self.gamma * self.np_random.standard_normal()
        prev_S = self.S
        self.S += self.alpha + self.sigma * self.np_random.standard_normal()
        self.Q_rem -= x
        self.T_rem -= 1

        # Benchmark
        # bench_action = self.target_q / self.T

        observation = self._get_obs()
        info = self._get_info()
        
        reward = -x * observation[1] + (self.target_q - self.Q_rem - x) * (self.S-prev_S)

        # An episode is terminated when there's no more time
        terminated = True if self.T_rem <= 0 else False

        return observation, reward, terminated, False, info
