import numpy as np
import gymnasium as gym
from gymnasium import spaces, logger


class TradingEnv(gym.Env):
    
    metadata = {}

    def __init__(self,
                 render_mode=None,
                 T = 100., # total trading period
                 lamb = 0.01, # magnitude of price impact
                 gamma = 0.02, # std dev of alpha signal
                 sigma = 0.1, # asset volatility
                 c = 0.5, # concavity
                 impact_decay = 10., # impact decay
                 theta = 5, # alpha decay
                 init_q = 1000, # initial position
                 target_q = 0, # target quantity
                 S0 = 10, # initial price of asset
                 ):
        
        assert T > 0, "T must be positive"
        assert lamb > 0, "lamb must be positive"
        assert gamma > 0, "gamma must be positive"
        assert sigma > 0, "sigma must be positive"
        assert 0 < c <= 1, "c must be in (0,1]"
        assert impact_decay > 0, "impact decay must be positive"
        assert theta>0, "alpha decay must be positive"
        assert target_q != init_q, "init_q cannot be same as target_q"
        assert S0 > 0, "Initial price must be positive"

        self.T = T
        self.T_rem = None
        self.lamb = lamb
        self.gamma = gamma
        self.sigma = sigma
        self.c = c
        self.impact_decay = impact_decay
        self.theta = theta
        self.init_q = init_q
        self.target_q = target_q
        self.S0 = S0
        self.MAX_J = abs(target_q - init_q)
        self.A = 1/(1-np.exp(-2/self.theta)) # for calculating std of S

        # self.bench_x = self.target_q / self.

        # for low and high, state variables are J, alpha, S, Q_rem, T_rem
        low = np.array([-1, -np.inf, -np.inf, -1, 0])
        high = np.array([0, np.inf, np.inf, 0, 1])
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float64)

        # Actions will be rescaled in step function
        self.action_space = spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.float32)

    # # count aggregates the number of samples seen so far
    # def update(self, new_value):
    #     (count, mean, M2) = self.P_stats
    #     count += 1
    #     delta = new_value - mean
    #     mean += delta / count
    #     delta2 = new_value - mean
    #     M2 += delta * delta2
    #     return (count, mean, M2)

    # # Retrieve the mean, std dev and sample std dev from an aggregate
    # def finalize(self):
    #     (count, mean, M2) = self.P_stats
    #     if count < 2:
    #         return (mean, 1, 1)
    #     else:
    #         (mean, stddev, sample_stddev) = (mean, np.sqrt(M2 / count), np.sqrt(M2 / (count - 1)))
    #         return (mean, stddev, sample_stddev)
        
    def _get_obs(self):
        norm_J = self.J/self.MAX_J # *2-1
        t = self.T - self.T_rem
        alpha_std = self.gamma*np.sqrt((1-np.exp(-2*(t+1)/self.theta))/(1-np.exp(-2/self.theta)))
        norm_alpha = self.alpha/alpha_std
        S_var = t*self.sigma**2+self.gamma**2*self.A*(t-self.A*np.exp(-2/self.theta)*(1-np.exp(-2*t/self.theta)))
        S_var += 2*self.gamma**2/((1-np.exp(2/self.theta))*(1-np.exp(-1/self.theta)))*(np.exp(-1/self.theta)*self.A*(1-np.exp(-2*(t-1)/self.theta))-np.exp(-t/self.theta)*(1-np.exp(-(t-1)/self.theta))/(1-np.exp(-1/self.theta))-(t-1)*np.exp(1/self.theta)+np.exp((-t+2)/self.theta)*(1-np.exp((t-1)/self.theta))/(1-np.exp(1/self.theta)))
        S_std = np.sqrt(S_var)
        if t == 0:
            norm_S = 0
        else:
            norm_S = (self.S-self.S0)/S_std
        norm_Q_rem = -self.Q_rem/(self.target_q - self.init_q)
        norm_T_rem = self.T_rem/self.T
        return np.array([norm_J, norm_alpha, norm_S, norm_Q_rem, norm_T_rem], dtype = np.float64)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.J = 0.
        self.alpha = self.np_random.normal(scale = self.gamma) # forecast
        self.S = self.S0
        self.Q_rem = self.target_q - self.init_q # position
        self.T_rem = self.T

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
        if self.c == 0.5:
            I = self.lamb * np.sign(self.J) * np.sqrt(abs(self.J))
        else:
            I = self.lamb * np.sign(self.J) * abs(self.J) ** self.c
        P = self.S + I
        
        # Calculate for next time step
        prev_S = self.S
        self.S += self.alpha + self.np_random.normal(scale = self.sigma)
        self.Q_rem -= x
        reward = -x * I + (self.target_q - self.Q_rem) * (self.S-prev_S)
        self.alpha = np.exp(-1/self.theta) * self.alpha + self.np_random.normal(scale = self.gamma)
        self.T_rem -= 1

        # Benchmark
        # bench_action = self.target_q / self.T

        observation = self._get_obs()
        info = self._get_info()
        
        # An episode is terminated when there's no more time
        terminated = True if self.T_rem <= 0 or self.Q_rem >= 0 else False

        return observation, reward, terminated, False, info
