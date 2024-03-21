import optuna
import logging
import sys
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeAwareObservation
import argparse
import os
import ST_tradingenv
import utils
import TD3
import OurDDPG
import DDPG
import time
from functools import partial

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env = TimeAwareObservation(eval_env)
    eval_env.reset(seed = seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset()[0], False

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)

            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes
    return avg_reward

def objective(trial, args):
    env_kwargs = {
    # "T": args.T,
    "tau": args.impact_decay,
    "lamb": args.lamb,
    "gamma": 0.01,
    "sigma": 0.3,
    "c": 0.5,
    "init_pos": args.init_position,
    "end_pos": args.end_position,
    "S0": 10.
    }
    
    env = gym.make(args.env, **env_kwargs)
    env = TimeAwareObservation(env)

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": trial.suggest_float("tau", 1e-4, 0.1, log = True),
        "total_q": max(args.init_position, args.end_position)
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = trial.suggest_float("policy_noise", 1e-4, 0.5, log = True) * max_action
        kwargs["noise_clip"] = trial.suggest_float("noise_clip", 1e-4, 0.9, log = True) * max_action
        kwargs["policy_freq"] = trial.suggest_int("policy_freq", 2, 100, step = 5)
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    state, done = env.reset(seed = args.seed)[0], False

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    q_rem = max(args.init_position, args.end_position)

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.np_random.uniform(low = 0, high = max_action, size = action_dim)
        else:
            action = (
                policy.select_action(np.array(state))
                + env.np_random.normal(0, max_action * trial.suggest_float("expl_noise", 1e-4, 0.1, log = True), size=action_dim)
            ).clip(0, max_action)
            # print(policy.select_action(np.array(state)))
            # print(env.np_random.normal(0, q_rem * args.expl_noise, size=action_dim))

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # print_bad_obs(action, next_state)
        q_rem -= action[0]
        done = terminated or truncated
        done_bool = float(done)#  if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        # print(t, args.start_timesteps)
        # print(f"State: {state}")
        # print(f"Action: {action}")
        # print(f"Next state: {next_state}")
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            trial.report(eval_policy(policy, args.env, args.seed), t)
        
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
            state, done = env.reset()[0], False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            q_rem = max(args.init_position, args.end_position)

        # if (t + 1) % args.eval_freq == 0:
        #     if args.save_model: policy.save(f"./models/{file_name}")

    return eval_policy(policy, args.env, args.seed)


if __name__ == "__main__":
    
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default='ST_tradingenv/Trading-v0')# Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gymnasium, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e2, type=int)# Time steps initial random policy is used
    # parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--T", default=100)
    parser.add_argument("--impact_decay", default=10)
    parser.add_argument("--lamb", default=0.001)
    parser.add_argument("--init_position", default=0)
    parser.add_argument("--end_position", default=1000)
    
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env.rsplit('/', 1)[-1]}_{args.seed}"

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    objective = partial(objective, args = args)
    study.optimize(objective, n_trials=100, n_jobs = 3)

    # if not os.path.exists("./results"):
    # 	os.makedirs("./results")

    # if args.save_model and not os.path.exists("./models"):
    # 	os.makedirs("./models")
                
    
