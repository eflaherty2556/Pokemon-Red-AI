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
	if len(sys.argv) != 5:
		print("Invalid args!\n")
		print("Use: Python3 hyperparameter_tuner.py base_model_name training_steps evaluation_steps hyperparameter_set\n")
		print("(use base_model_name=None for no tensorboard logging)\n")
		sys.exit()

	base_model_name = sys.argv[1]
	training_steps = int(sys.argv[2])
	evaluation_steps = int(sys.argv[3])
	hyperparameter_set = int(sys.argv[4])

	if hyperparameter_set < 1 or hyperparameter_set > 3:
		print("unknown hyperparameter set selected. Please use sets 1, 2, or 3")
		sys.exit()

	retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))

	print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
	env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE, use_restricted_actions=retro.Actions.ALL)
	env = Discretizer(env)
	
	#Wrap enviornment for skip limit
	env = SkipLimit(env=env, time_between_steps=3)    

	vec_env = make_vec_env(lambda: env, n_envs=32)
	vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

	test_env = make_vec_env(lambda: env, n_envs=1) # only one env for speed
	test_env = VecNormalize(test_env, norm_obs=True, norm_reward=True)

	# hyperparameters to tune ((1-2) * 3 * 3 * 2 * 3)(3 * 3 * 3)(3 * 4 * 3)

	# model (set 1)
	# policy = [MlpPolicy]
	policy = [MlpPolicy, CnnPolicy]
	gamma = [0.982, 0.990, 0.997]
	learning_rate = [0.00035, 0.0007, 0.0012]
	n_steps = [5, 10]
	schedule = ["linear", "constant", "middle_drop"]
	# loss calc (set 2)
	value_function_coefficient = [0.20, 0.25, 0.30]
	entropy_coefficient = [0.005, 0.01, 0.05]
	max_gradient_norm = [0.45, 0.50, 0.55]
	# RMSProp (set 3)
	alpha = [0.98, 0.99, 0.999]
	momentum = [0.0, 1e-05, 1e-03, 1e-02]
	epsilon = [1e-06, 1e-05, 1e-04]
	
	performances = []
	test_number = 0
	toal_tests = 1

	if hyperparameter_set == 1:
		write_to_results(hyperparameter_set, "\n\nTraining on hyperparameter set 1...\n")
		total_tests = len(policy) * len(gamma) * len(learning_rate) * len(n_steps) * len(schedule)
		for p in policy:
			for g in gamma:
				for l in learning_rate:
					for n in n_steps:
						for s in schedule:
							test_number += 1
							vec_env.reset()
							model = A2C(p, vec_env, gamma=g, learning_rate=l, n_steps=n, lr_schedule=s, verbose=1, tensorboard_log="./pokemon-red-tensorboard/")

							if base_model_name is not None:
								model_name = base_model_name + "-" + str(p) + "-" + str(g) + "-" + str(l) + "-" + str(n) + "-" + str(s)
								model.learn(total_timesteps=training_steps, tb_log_name=model_name)
							else:
								model.learn(total_timesteps=training_steps, tb_log_name=None)

							write_to_results(hyperparameter_set, "Finished training, now evaluating\n")
							reward_per_step, cumulative_reward = evaluate(model, evaluation_steps, test_env)

							performance = [reward_per_step, cumulative_reward, p, g, l, n, s]
							performances.append(performance)
							write_to_results(hyperparameter_set, "\nFinished test " + str(test_number) + "/" + str(total_tests) + "\n")
							write_to_results(hyperparameter_set, "Configuration: \n")
							write_to_results(hyperparameter_set, "p = " + str(p) + "; g = " + str(g) + "; l = " + str(l) + "; n = " + str(n) + "; s = " + str(s) + ";\n")
							write_to_results(hyperparameter_set, "Result: \n")
							write_to_results(hyperparameter_set, "reward_per_step = " + str(reward_per_step) + "; cumulative_reward = " + str(cumulative_reward) + ";\n")

		write_to_results(hyperparameter_set, "DONE!\n")
		performances.sort(key=lambda x: x[0])
		write_to_results(hyperparameter_set, "Best set 1 configuration (by reward per step): " + str(performances[0]) + "\n")

	elif hyperparameter_set == 2:
		write_to_results(hyperparameter_set, "\n\nTraining on hyperparameter set 2...")
		total_tests = len(value_function_coefficient) * len(entropy_coefficient) * len(max_gradient_norm)
		for v in value_function_coefficient:
			for e in entropy_coefficient:
				for m in max_gradient_norm:
					test_number += 1
					vec_env.reset()
					model = A2C(MlpPolicy, vec_env, vf_coef=v, ent_coef=e, max_grad_norm=m, verbose=1, tensorboard_log="./pokemon-red-tensorboard/")

					if base_model_name  is not None:
						model_name = base_model_name + "-" + str(v) + "-" + str(e) + "-" + str(m)
						model.learn(total_timesteps=training_steps, tb_log_name=model_name)
					else:
						model.learn(total_timesteps=training_steps, tb_log_name=None)

					write_to_results(hyperparameter_set, "Finished training, now evaluating\n")
					reward_per_step, cumulative_reward = evaluate(model, evaluation_steps, test_env)

					performance = [reward_per_step, cumulative_reward, v, e, m]
					performances.append(performance)
					write_to_results(hyperparameter_set, "\nFinished test " + str(test_number) + "/" + str(total_tests) + "\n")
					write_to_results(hyperparameter_set, "Configuration: \n")
					write_to_results(hyperparameter_set, "v = " + str(v) + "; e = " + str(e) + "; m = " + str(m) + ";\n")
					write_to_results(hyperparameter_set, "Result: \n")
					write_to_results(hyperparameter_set, "reward_per_step = " + str(reward_per_step) + "; cumulative_reward = " + str(cumulative_reward) + ";\n")

		write_to_results(hyperparameter_set, "DONE!\n")
		performances.sort(key=lambda x: x[0])
		write_to_results(hyperparameter_set, "Best set 2 configuration (by reward per step): " + str(performances[0]) + "\n")

	elif hyperparameter_set == 3:
		write_to_results(hyperparameter_set, "\n\nTraining on hyperparameter set 3...")
		total_tests = len(alpha) * len(momentum) * len(epsilon)
		for a in alpha:
			for m in momentum:
				for e in epsilon:
					test_number += 1
					vec_env.reset()
					model = A2C(MlpPolicy, vec_env, alpha=a, momentum=m, epsilon=e, verbose=1, tensorboard_log="./pokemon-red-tensorboard/")

					if base_model_name is not None:
						model_name = base_model_name + "-" + str(a) + "-" + str(m) + "-" + str(e)
						model.learn(total_timesteps=training_steps, tb_log_name=model_name)
					else:
						model.learn(total_timesteps=training_steps, tb_log_name=None)

					write_to_results(hyperparameter_set, "Finished training, now evaluating\n")
					reward_per_step, cumulative_reward = evaluate(model, evaluation_steps, test_env)

					performance = [reward_per_step, cumulative_reward, a, m, e]
					performances.append(performance)
					write_to_results(hyperparameter_set, "\nFinished test " + str(test_number) + "/" + str(total_tests) + "\n")
					write_to_results(hyperparameter_set, "Configuration: \n")
					write_to_results(hyperparameter_set, "a = " + str(a) + "; m = " + str(m) + "; e = " + str(e) + ";\n")
					write_to_results(hyperparameter_set, "Result: \n")
					write_to_results(hyperparameter_set, "reward_per_step = " + str(reward_per_step) + "; cumulative_reward = " + str(cumulative_reward) + ";\n")

		write_to_results(hyperparameter_set, "DONE!\n")
		performances.sort(key=lambda x: x[0])
		write_to_results(hyperparameter_set, "Best set 3 configuration (by reward per step): " + str(performances[0]) + "\n")

def evaluate(model, evaluation_steps, env):
	cumulative_reward = 0

	obs = env.reset()
	for i in range(evaluation_steps):
		action, _states = model.predict(obs)
		obs, reward, done, info = env.step(action)
		cumulative_reward += max(reward) 
		# get max reward because of vec env

	reward_per_step = cumulative_reward/evaluation_steps

	return reward_per_step, cumulative_reward

def write_to_results(hyperparameter_set, text_to_write):
	print(text_to_write) # print text as well as writing to file

	filename = "hyperparameters_" + str(hyperparameter_set) + ".txt"
	with open(filename, "a") as file:
		file.write(text_to_write)

if __name__ == "__main__":
		main()
