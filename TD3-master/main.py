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


def print_bad_obs(action, state):
	assert len(state) == 6
	if state[4] > 100:
		print("Time:")
		print(state)
		print(f"action: {action}")
	if state[5] > 1000:
		print("Position:")
		print(state)
		print(f"action: {action}")

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
			print_bad_obs(action, state)
			done = terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	start_time = time.time()

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default='ST_tradingenv/Trading-v0')# Gymnasium environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gymnasium, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
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

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

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
	# print(env.reset(seed = args.seed))
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	# np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"total_q": max(args.init_position, args.end_position)
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	# def objective(config):  # ①
	# 	kwargs = {
	# 		"state_dim": state_dim,
	# 		"action_dim": action_dim,
	# 		"max_action": max_action,
	# 		"discount": args.discount,
	# 		"tau": config["tau"],
	# 		"total_q": max(args.init_position, args.end_position)
	# 	}
	# 	kwargs["policy_noise"] = config["policy_noise"] * max_action
	# 	kwargs["noise_clip"] = config["noise_clip"] * max_action
	# 	kwargs["policy_freq"] = config["policy_freq"]
	# 	kwargs["actor_lr"] = config["actor_lr"]
	# 	kwargs["critic_lr"] = config["critic_lr"]
	# 	policy = TD3.TD3(**kwargs)

	# 	while True:
	# 		evaluations = [eval_policy(policy, args.env, args.seed)]
	# 		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# 		state, done = env.reset(seed = args.seed)[0], False

	# 		episode_reward = 0
	# 		episode_timesteps = 0
	# 		episode_num = 0
	# 		q_rem = max(args.init_position, args.end_position)

	# 		for t in range(int(args.max_timesteps)):
				
	# 			episode_timesteps += 1

	# 			# Select action randomly or according to policy
	# 			if t < args.start_timesteps:
	# 				action = env.np_random.uniform(low = 0, high = max_action, size = action_dim)
	# 			else:
	# 				action = (
	# 					policy.select_action(np.array(state))
	# 					+ env.np_random.normal(0, max_action * args.expl_noise, size=action_dim)
	# 				).clip(0, max_action)
	# 				# print(policy.select_action(np.array(state)))
	# 				# print(env.np_random.normal(0, q_rem * args.expl_noise, size=action_dim))

	# 			# Perform action
	# 			next_state, reward, terminated, truncated, _ = env.step(action)
	# 			# print_bad_obs(action, next_state)
	# 			q_rem -= action[0]
	# 			done = terminated or truncated
	# 			done_bool = float(done) # if episode_timesteps < env._max_episode_steps else 0

	# 			# Store data in replay buffer
	# 			# print(t, args.start_timesteps)
	# 			# print(f"State: {state}")
	# 			# print(f"Action: {action}")
	# 			# print(f"Next state: {next_state}")
	# 			replay_buffer.add(state, action, next_state, reward, done_bool)

	# 			state = next_state
	# 			episode_reward += reward

	# 			# Train agent after collecting sufficient data
	# 			if t >= args.start_timesteps:
	# 				policy.train(replay_buffer, args.batch_size)

	# 			if done: 
	# 				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
	# 				# print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
	# 				# Reset environment
	# 				state, done = env.reset()[0], False
	# 				episode_reward = 0
	# 				episode_timesteps = 0
	# 				episode_num += 1
	# 				q_rem = max(args.init_position, args.end_position)

	# 			# Evaluate episode
	# 			if (t + 1) % args.eval_freq == 0:
	# 				evaluations.append(eval_policy(policy, args.env, args.seed))
	# 				np.save(f"./results/{file_name}", evaluations)
	# 				if args.save_model: policy.save(f"./models/{file_name}")
	# 		acc = evaluations[-1]  # Compute test accuracy
	# 		train.report({"reward": acc})  # Report to Tune


	# search_space = {
	# 	"tau": tune.loguniform(1e-4, 1e-2),
	# 	"policy_noise": tune.loguniform(1e-4, 0.5),
	# 	"noise_clip": tune.loguniform(1e-4, 0.9),
	# 	"policy_freq": tune.uniform(2, 100),
	# 	"actor_lr": tune.loguniform(1e-5, 0.1),
	# 	"critic_lr": tune.loguniform(1e-5, 0.1)
	# 	}
	# algo = OptunaSearch()  # ②

	# tuner = tune.Tuner(  # ③
	# 	objective,
	# 	tune_config=tune.TuneConfig(
	# 		metric="mean_accuracy",
	# 		mode="max",
	# 		search_alg=algo,
	# 	),
	# 	run_config=train.RunConfig(
	# 		stop={"training_iteration": 5},
	# 	),
	# 	param_space=search_space,
	# )
	# results = tuner.fit()
	# print("Best config is:", results.get_best_result().config)




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
				+ env.np_random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(0, max_action)
			# print(policy.select_action(np.array(state)))
			# print(env.np_random.normal(0, q_rem * args.expl_noise, size=action_dim))

		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action)
		# print_bad_obs(action, next_state)
		q_rem -= action[0]
		done = terminated or truncated
		done_bool = float(done) # if episode_timesteps < env._max_episode_steps else 0

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
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset()[0], False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			q_rem = max(args.init_position, args.end_position)

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
	
	print(f"Total time for {args.policy}: {time.time()-start_time}")
