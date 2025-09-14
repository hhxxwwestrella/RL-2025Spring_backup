import gymnasium as gym
from typing import List, Tuple

from rl2025.exercise4.agents import DDPG
from rl2025.exercise4.evaluate_ddpg import evaluate
from rl2025.exercise5.train_ddpg import RACETRACK_CONFIG

RENDER = True

CONFIG = RACETRACK_CONFIG
CONFIG['save_filename'] = "ModelB.pt" # or "ModelB.pt"

if __name__ == "__main__":
    env = gym.make(CONFIG["env"], render_mode="human")
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
