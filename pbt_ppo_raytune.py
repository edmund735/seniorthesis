import gymnasium as gym
from gymnasium import spaces, logger
import random
import ray
from ray import train, tune
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
import argparse
import numpy as np
import os
import argparse
import pandas as pd
from datetime import datetime
from trading_env_rllib import TradingEnv

# from ray.tune import run, sample_from
# from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function

def env_creator(env_config):
    return TradingEnv(env_config)

register_env("Trading-v0", env_creator)

# # Postprocess the perturbed config to ensure it's still valid used if PBT.
# def explore(config):
#     # Ensure we collect enough timesteps to do sgd.
#     if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
#         config["train_batch_size"] = config["sgd_minibatch_size"] * 2
#     # Ensure we run at least one sgd iter.
#     if config["lambda"] > 1:
#         config["lambda"] = 1
#     config["train_batch_size"] = int(config["train_batch_size"])
#     return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test

    perturbation_interval = 5
    pbt = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds={
            # hyperparameter bounds.
            "lr": [0.0001, 0.02],
        },
    )

    tuner = tune.Tuner(
        pbt_function,
        run_config=train.RunConfig(
            name="pbt_test",
            verbose=True,
            stop={
                "training_iteration": 30,
            },
            failure_config=train.FailureConfig(
                fail_fast=True,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            metric="mean_accuracy",
            mode="max",
            num_samples=8,
            reuse_actors=True,
        ),
        param_space={
            "lr": 0.0001,
            # note: this parameter is perturbed but has no effect on
            # the model training in this example
            "some_other_factor": 1,
            # This parameter is not perturbed and is used to determine
            # checkpoint frequency. We set checkpoints and perturbations
            # to happen at the same frequency.
            "checkpoint_interval": perturbation_interval,
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max", type=int, default=1000000)
#     parser.add_argument("--algo", type=str, default="PPO")
#     parser.add_argument("--num_workers", type=int, default=3)
#     parser.add_argument("--num_samples", type=int, default=4)
#     parser.add_argument("--t_ready", type=int, default=50000)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument(
#         "--horizon", type=int, default=1600
#     )  # make this 1000 for other envs
#     parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
#     parser.add_argument("--env_name", type=str, default="Pendulum-v1")
#     parser.add_argument(
#         "--criteria", type=str, default="timesteps_total"
#     )  # "training_iteration", "time_total_s"
#     parser.add_argument(
#         "--net", type=str, default="32_32"
#     )  # May be important to use a larger network for bigger tasks.
#     parser.add_argument("--filename", type=str, default="")
#     parser.add_argument("--method", type=str, default="pb2")  # ['pbt', 'pb2']
#     parser.add_argument("--save_csv", type=bool, default=False)

#     args = parser.parse_args()

#     # bipedalwalker needs 1600
#     if args.env_name in ["BipedalWalker-v2", "BipedalWalker-v3"]:
#         horizon = 1600
#     else:
#         horizon = 1000

#     pbt = PopulationBasedTraining(
#         time_attr=args.criteria,
#         metric="episode_reward_mean",
#         mode="max",
#         perturbation_interval=args.t_ready,
#         resample_probability=args.perturb,
#         quantile_fraction=args.perturb,  # copy bottom % with top %
#         # Specifies the search space for these hyperparams
#         hyperparam_mutations={
#             "lambda": lambda: random.uniform(0.9, 1.0),
#             "clip_param": lambda: random.uniform(0.1, 0.5),
#             "lr": lambda: random.uniform(1e-3, 1e-5),
#             "train_batch_size": lambda: random.randint(1000, 60000),
#         },
#         custom_explore_fn=explore,
#     )

#     pb2 = PB2(
#         time_attr=args.criteria,
#         metric="episode_reward_mean",
#         mode="max",
#         perturbation_interval=args.t_ready,
#         quantile_fraction=args.perturb,  # copy bottom % with top %
#         # Specifies the hyperparam search space
#         hyperparam_bounds={
#             "lambda": [0.9, 1.0],
#             "clip_param": [0.1, 0.5],
#             "lr": [1e-5, 1e-3],
#             "train_batch_size": [1000, 60000],
#         },
#     )

#     methods = {"pbt": pbt, "pb2": pb2}

#     timelog = (
#         str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
#     )

#     args.dir = "{}_{}_{}_Size{}_{}_{}".format(
#         args.algo,
#         args.filename,
#         args.method,
#         str(args.num_samples),
#         args.env_name,
#         args.criteria,
#     )

#     analysis = run(
#         args.algo,
#         name="{}_{}_{}_seed{}_{}".format(
#             timelog, args.method, args.env_name, str(args.seed), args.filename
#         ),
#         scheduler=methods[args.method],
#         verbose=1,
#         num_samples=args.num_samples,
#         reuse_actors=True,
#         stop={args.criteria: args.max},
#         config={
#             "env": args.env_name,
#             "log_level": "INFO",
#             "seed": args.seed,
#             "kl_coeff": 1.0,
#             "num_gpus": 0,
#             "horizon": horizon,
#             "observation_filter": "MeanStdFilter",
#             "model": {
#                 "fcnet_hiddens": [
#                     int(args.net.split("_")[0]),
#                     int(args.net.split("_")[1]),
#                 ],
#                 "free_log_std": True,
#             },
#             "num_sgd_iter": 10,
#             "sgd_minibatch_size": 128,
#             "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
#             "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
#             "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
#             "train_batch_size": sample_from(lambda spec: random.randint(1000, 60000)),
#         },
#     )

#     all_dfs = list(analysis.trial_dataframes.values())

#     results = pd.DataFrame()
#     for i in range(args.num_samples):
#         df = all_dfs[i]
#         df = df[
#             [
#                 "timesteps_total",
#                 "episodes_total",
#                 "episode_reward_mean",
#                 "info/learner/default_policy/cur_kl_coeff",
#             ]
#         ]
#         df["Agent"] = i
#         results = pd.concat([results, df]).reset_index(drop=True)

#     if args.save_csv:
#         if not (os.path.exists("data/" + args.dir)):
#             os.makedirs("data/" + args.dir)

#         results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))