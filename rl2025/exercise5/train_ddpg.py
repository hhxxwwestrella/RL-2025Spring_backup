import copy
import pickle
from collections import defaultdict
import highway_env
import gymnasium as gym
from gymnasium import Space
import highway_env as hiv
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2025.constants import EX4_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS
from rl2025.exercise4.agents import DDPG
from rl2025.exercise3.replay import ReplayBuffer
from rl2025.util.hparam_sweeping import generate_hparam_configs
from rl2025.util.result_processing import Run

import random
import torch


RENDER = False
SWEEP = True
NUM_SEEDS_SWEEP = 1
SWEEP_SAVE_RESULTS = True
SWEEP_SAVE_ALL_WEIGTHS = False
ENV = "RACETRACK"
RACETRACK_CONFIG = {
    "critic_hidden_size": [512, 256, 128],
    "policy_hidden_size": [512, 256, 256],
}
RACETRACK_CONFIG.update(RACETRACK_CONSTANTS)

RACETRACK_HPARAMS = {
    "critic_hidden_size": ...,  # Placeholder
    "policy_hidden_size": ...,
}

SWEEP_RESULTS_FILE_RACETRACK = "DDPG-Racetrack-sweep-results-ex4.pkl"



from highway_env.road.lane import StraightLane, CircularLane


def place_blue_car_ahead(env, distance):
    """
    Place the blue car on the same lane, ahead of our yellow car.
    (Force our model to learn to avoid crashing)
    """
    yellow_car = env.vehicle
    for car in env.road.vehicles:
        if car is not yellow_car:
            blue_car = car
            break

    lane = env.road.network.get_lane(yellow_car.lane_index)
    if isinstance(lane, StraightLane):
        heading = yellow_car.heading
        offset = distance * np.array([np.cos(heading), np.sin(heading)])
        target_blue_car_position = yellow_car.position + offset
        blue_car.position = target_blue_car_position


def play_episode(
        env,
        agent,
        replay_buffer,
        timestep,
        train=True,
        explore=True,
        render=False,
        max_steps=200,
        batch_size=64,
):

    ep_data = defaultdict(list)
    obs, _ = env.reset()

    # Place the blue car ahead of the yellow one to
    if train: # max_timesteps = 31000
        distance = np.random.uniform(20, 30)
        place_blue_car_ahead(env, distance)
        current_timestep = timestep

    obs = obs.ravel()
    done = False
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, terminated, truncated, _ = env.step(action)

        # Penalise for crashing
        if env.vehicle.crashed:
            if train and current_timestep < 15000:
                reward -= 50.0
                print("Crashed. Reward-50. Episode terminated.")
            elif train and current_timestep < 31000:
                reward -= 100.0
                print("Crashed at later stage. Reward-100. Episode terminated.")

        nobs = nobs.ravel()
        done = terminated or truncated

        # Penalise the car for getting off the road
        if train and not env.vehicle.on_road:
            reward -= 30.0
            print("Got out of the road. Reward-30.")
            if current_timestep < 15000:
                done = True  # And end the episode

        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                new_data = agent.update(batch)
                for k, v in new_data.items():
                    ep_data[k].append(v)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        obs = nobs

    return episode_timesteps, episode_return, ep_data


def train(env: gym.Env, env_eval: gym.Env, config: Dict, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    obs, _ = env.reset()
    obs = obs.ravel()
    observation_space = Space((obs.shape[0],))

    timesteps_elapsed = 0

    agent = DDPG(
        action_space=env.action_space, observation_space=observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()

    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break

            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            episode_timesteps, ep_return, ep_data = play_episode(
                env,
                agent,
                replay_buffer,
                timestep=timesteps_elapsed,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                episodic_returns = []
                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env_eval,
                        agent,
                        replay_buffer,
                        timestep=timesteps_elapsed,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                    episodic_returns.append(episode_return)

                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}. {episodic_returns}\n"
                    )
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

                # if min(episodic_returns) >= config["target_return"]+500:
                #     pbar.write(
                #         f"Reached return {eval_returns} >= target return of {config['target_return']}"
                #     )
                #     break
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time}")

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":
    if ENV == "RACETRACK":
        CONFIG = RACETRACK_CONFIG
        HPARAMS_SWEEP = None
        SWEEP_RESULTS_FILE = None
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    env_eval = gym.make(CONFIG["env"])

    if SWEEP and HPARAMS_SWEEP is not None:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run...")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i+1}/{NUM_SEEDS_SWEEP}")
                run_save_filename = '--'.join([run.config["algo"], run.config["env"], hparams_values, str(i)])
                if SWEEP_SAVE_ALL_WEIGTHS:
                    run.set_save_filename(run_save_filename)
                eval_returns, eval_timesteps, times, run_data = train(env, run.config, output=False)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")

        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)

    else:
        _ = train(env, env_eval, CONFIG)

    env.close()
