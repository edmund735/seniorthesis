from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym
import ST_tradingenv

env = gym.make('ST_tradingenv/Trading-v0')
# reset the environment to get the first observation

class QLAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            m = np.ones(1001, dtype = np.int8)
            if obs[2] + 1 < len(m):
                m[obs[2]+1:] = 0
            return env.action_space.sample(mask = m)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)


# hyperparameters
learning_rate = 0.01
n_episodes = 100_00
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = QLAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

    rolling_length = 500

    
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()

state_value = defaultdict(float)
policy = defaultdict(int)
for obs, action_values in agent.q_values.items():
    state_value[obs] = float(np.max(action_values))
    policy[obs] = int(np.argmax(action_values))

# J_vals = np.zeros((len(state_value), 2))
I_vals = np.zeros((len(state_value), 2))
alpha_vals = np.zeros((len(state_value), 2))
X_vals = np.zeros((len(state_value), 2))
T_rem_vals = np.zeros((len(state_value), 2))
pi_vals = np.zeros((len(state_value), 2))

for i, k in enumerate(state_value.keys()):
    I_vals[i,0] = k[0]
    alpha_vals[i,0] = k[1]
    X_vals[i,0] = k[2]
    T_rem_vals[i,0] = k[3]
    pi_vals[i,0] = k[4]

    I_vals[i,1] = state_value[k]
    alpha_vals[i,1] = state_value[k]
    X_vals[i,1] = state_value[k]
    T_rem_vals[i,1] = state_value[k]
    pi_vals[i,1] = state_value[k]

    # Scatter plot for I_vals
plt.scatter(I_vals[:, 0], I_vals[:, 1])
plt.yscale('log')  # Set y-axis to log scale
plt.title("I")
plt.xlabel("I")
plt.show()

# Scatter plot for alpha_vals
plt.scatter(alpha_vals[:, 0], alpha_vals[:, 1])
plt.yscale('log')  # Set y-axis to log scale
plt.title("alpha")
plt.show()

# Scatter plot for X_vals
plt.scatter(X_vals[:, 0], X_vals[:, 1])
plt.yscale('log')  # Set y-axis to log scale
plt.title("X")
plt.show()

# Scatter plot for T_rem_vals
plt.scatter(T_rem_vals[:, 0], T_rem_vals[:, 1])
plt.yscale('log')  # Set y-axis to log scale
plt.title("T_rem")
plt.show()

# Scatter plot for pi_vals
plt.scatter(pi_vals[:, 0], pi_vals[:, 1])
plt.yscale('log')  # Set y-axis to log scale
plt.title("pi")
plt.show()