import retro
import gym
import os
import time
import retro.enums
import sys

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecNormalize
from skipWrapper import SkipLimit

from Discretizer import Discretizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
	if len(sys.argv) != 3:
		print("Invalid args!\n")
		print("Use: Python3 model_player.py model_dir reward_log_frequency\n")
		sys.exit()

	model_dir = sys.argv[1]
	reward_log_frequency = sys.argv[2]

	retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))

	print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
	env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE, use_restricted_actions=retro.Actions.ALL)
	env = Discretizer(env)
	
	#Wrap enviornment for skip limit
	env = SkipLimit(env=env, time_between_steps=3)    

	vec_env = make_vec_env(lambda: env, n_envs=32)
	vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

	model = A2C.load(model_dir)
	obs = env.reset()

	cumulative_reward = 0
	timesteps = 0

	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()

		cumulative_reward += reward
		timesteps += 1

		if timesteps % reward_log_frequency == 0:
			print("\nCumulative Reward: ", cumulative_reward)
			print("Reward Per Step:", (cumulative_reward/timesteps))

	env.close()

	
if __name__ == "__main__":
		main()